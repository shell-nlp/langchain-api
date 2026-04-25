import hashlib
import json
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

from langchain_api.elastic_utils import Elasticsearch
from langchain_api.utils import get_chat_model


TRIPLET_PROMPT = """从文本中抽取知识图谱三元组。

要求：
- 只抽取文本明确表达的事实，不要补充常识。
- subject/object 使用简洁实体名。
- predicate 使用简短中文或英文关系短语。
- 最多返回 20 个三元组。
- 只返回 JSON，格式：{"triplets":[{"subject":"...","predicate":"...","object":"..."}]}

文本：
{text}
"""

QUERY_ENTITY_PROMPT = """从问题中抽取检索知识图谱需要的实体名。
只返回 JSON：{"entities":["..."]}

问题：
{query}
"""


class ElasticGraphRAG:
    """基于 Elasticsearch 的轻量 Vector Graph RAG。"""

    def __init__(
        self,
        es: Optional[Elasticsearch],
        graph_name: str,
        chat_model=None,
    ):
        self.es = es
        self.graph_name = graph_name
        self.chat_model = chat_model
        self.indexes = self.index_names(graph_name)

    @staticmethod
    def index_names(prefix: str) -> Dict[str, str]:
        return {
            "passage": f"{prefix}_passages",
            "entity": f"{prefix}_entities",
            "relation": f"{prefix}_relations",
        }

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        extract_triplets: bool = True,
    ) -> Dict[str, Any]:
        documents = []
        for index, text in enumerate(texts):
            metadata = metadatas[index] if metadatas and index < len(metadatas) else {}
            doc_id = ids[index] if ids and index < len(ids) else str(uuid.uuid4())
            documents.append(Document(page_content=text, metadata=metadata, id=doc_id))

        return self.add_documents(documents, extract_triplets=extract_triplets)

    def add_documents(
        self,
        documents: List[Document],
        extract_triplets: bool = True,
    ) -> Dict[str, Any]:
        """
        把普通文档转成 ES 向量图 RAG 需要的三类索引。

        推荐输入：
        - 普通 Document：自动用 LLM 抽取 triplets。
        - 预抽取 Document：metadata["triplets"] = [[subject, predicate, object], ...]
        """
        graph = self.build_graph(documents, extract_triplets=extract_triplets)

        self.bulk_upsert(self.indexes["entity"], graph["entities"])
        self.bulk_upsert(self.indexes["relation"], graph["relations"])
        self.bulk_upsert(self.indexes["passage"], graph["passages"])

        result = {
            "graph_name": self.graph_name,
            "indexes": self.indexes,
            "passage_count": len(graph["passages"]),
            "entity_count": len(graph["entities"]),
            "relation_count": len(graph["relations"]),
        }
        logger.info("ES向量图索引完成: {}", result)
        return result

    def retrieve(
        self,
        query: str,
        k: int = 6,
        entity_top_k: int = 5,
        relation_top_k: int = 8,
        expansion_degree: int = 1,
        relation_limit: int = 30,
        return_debug: bool = False,
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        query_entities = self.extract_query_entities(query)
        return self.es.vector_graph_retrieve(
            query=query,
            k=k,
            index_name=self.indexes["passage"],
            entity_index_name=self.indexes["entity"],
            relation_index_name=self.indexes["relation"],
            entity_top_k=entity_top_k,
            relation_top_k=relation_top_k,
            expansion_degree=expansion_degree,
            relation_limit=relation_limit,
            query_entities=query_entities,
            return_debug=return_debug,
        )

    def build_graph(
        self, documents: List[Document], extract_triplets: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        entity_name_to_id: Dict[str, str] = {}
        relation_text_to_id: Dict[str, str] = {}
        entity_to_relation_ids: Dict[str, set] = defaultdict(set)
        entity_to_passage_ids: Dict[str, set] = defaultdict(set)
        relation_to_entity_ids: Dict[str, set] = defaultdict(set)
        relation_to_passage_ids: Dict[str, set] = defaultdict(set)
        passage_to_entity_ids: Dict[str, set] = defaultdict(set)
        passage_to_relation_ids: Dict[str, set] = defaultdict(set)
        relation_triplets: Dict[str, Tuple[str, str, str]] = {}

        for document in documents:
            passage_id = str(document.id or uuid.uuid4())
            document.id = passage_id
            triplets = self.get_document_triplets(
                document, extract_triplets=extract_triplets
            )

            for subject, predicate, object_ in triplets:
                subject_id = self.get_entity_id(subject, entity_name_to_id)
                object_id = self.get_entity_id(object_, entity_name_to_id)
                relation_text = f"{subject} {predicate} {object_}"
                relation_id = relation_text_to_id.setdefault(
                    self.normalize(relation_text), self.stable_id("rel", relation_text)
                )

                relation_triplets[relation_id] = (subject, predicate, object_)
                relation_to_entity_ids[relation_id].update([subject_id, object_id])
                relation_to_passage_ids[relation_id].add(passage_id)
                entity_to_relation_ids[subject_id].add(relation_id)
                entity_to_relation_ids[object_id].add(relation_id)
                entity_to_passage_ids[subject_id].add(passage_id)
                entity_to_passage_ids[object_id].add(passage_id)
                passage_to_entity_ids[passage_id].update([subject_id, object_id])
                passage_to_relation_ids[passage_id].add(relation_id)

        entities = self.build_entity_docs(
            entity_name_to_id, entity_to_relation_ids, entity_to_passage_ids
        )
        relations = self.build_relation_docs(
            relation_triplets, relation_to_entity_ids, relation_to_passage_ids
        )
        passages = self.build_passage_docs(
            documents, passage_to_entity_ids, passage_to_relation_ids
        )
        return {"entities": entities, "relations": relations, "passages": passages}

    def build_entity_docs(
        self,
        entity_name_to_id: Dict[str, str],
        entity_to_relation_ids: Dict[str, set],
        entity_to_passage_ids: Dict[str, set],
    ) -> List[Dict[str, Any]]:
        docs = []
        id_to_entity_name = {
            entity_id: name for name, entity_id in entity_name_to_id.items()
        }
        for entity_id, entity_name in id_to_entity_name.items():
            docs.append(
                {
                    "id": entity_id,
                    "content": entity_name,
                    "metadata": {
                        "id": entity_id,
                        "name": entity_name,
                        "type": "entity",
                        "relation_ids": sorted(entity_to_relation_ids[entity_id]),
                        "passage_ids": sorted(entity_to_passage_ids[entity_id]),
                    },
                }
            )
        return docs

    def build_relation_docs(
        self,
        relation_triplets: Dict[str, Tuple[str, str, str]],
        relation_to_entity_ids: Dict[str, set],
        relation_to_passage_ids: Dict[str, set],
    ) -> List[Dict[str, Any]]:
        docs = []
        for relation_id, triplet in relation_triplets.items():
            subject, predicate, object_ = triplet
            docs.append(
                {
                    "id": relation_id,
                    "content": f"{subject} {predicate} {object_}",
                    "metadata": {
                        "id": relation_id,
                        "type": "relation",
                        "entity_ids": sorted(relation_to_entity_ids[relation_id]),
                        "passage_ids": sorted(relation_to_passage_ids[relation_id]),
                        "subject": subject,
                        "predicate": predicate,
                        "object": object_,
                    },
                }
            )
        return docs

    def build_passage_docs(
        self,
        documents: List[Document],
        passage_to_entity_ids: Dict[str, set],
        passage_to_relation_ids: Dict[str, set],
    ) -> List[Dict[str, Any]]:
        docs = []
        for document in documents:
            passage_id = str(document.id)
            metadata = dict(document.metadata or {})
            metadata.update(
                {
                    "id": passage_id,
                    "type": "passage",
                    "entity_ids": sorted(passage_to_entity_ids[passage_id]),
                    "relation_ids": sorted(passage_to_relation_ids[passage_id]),
                }
            )
            docs.append(
                {
                    "id": passage_id,
                    "content": document.page_content,
                    "metadata": metadata,
                }
            )
        return docs

    def get_document_triplets(
        self, document: Document, extract_triplets: bool = True
    ) -> List[Tuple[str, str, str]]:
        raw_triplets = document.metadata.get("triplets") if document.metadata else None
        if raw_triplets:
            return self.parse_triplets(raw_triplets)
        if not extract_triplets:
            return []
        return self.extract_triplets(document.page_content)

    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        model = self.get_chat_model()
        response = model.invoke(TRIPLET_PROMPT.format(text=text[:8000]))
        content = response.content if hasattr(response, "content") else str(response)
        return self.parse_triplets(self.load_json_object(content).get("triplets", []))

    def extract_query_entities(self, query: str) -> List[str]:
        try:
            model = self.get_chat_model()
            response = model.invoke(QUERY_ENTITY_PROMPT.format(query=query))
            content = response.content if hasattr(response, "content") else str(response)
            entities = self.load_json_object(content).get("entities", [])
            return [str(entity).strip() for entity in entities if str(entity).strip()]
        except Exception as exc:
            logger.warning("查询实体抽取失败，使用简单切词: {}", exc)
            return self.simple_extract_entities(query)

    def bulk_upsert(self, index_name: str, docs: List[Dict[str, Any]]) -> None:
        if not docs:
            return

        first_embedding = self.es.embedding_model.embed_query(docs[0]["content"])
        self.ensure_index(index_name, len(first_embedding))

        operations = []
        for index, doc in enumerate(docs):
            embedding = first_embedding if index == 0 else self.es.embedding_model.embed_query(doc["content"])
            operations.append({"index": {"_index": index_name, "_id": doc["id"]}})
            operations.append(
                {
                    "content": doc["content"],
                    "embedding": embedding,
                    "metadata": doc.get("metadata", {}),
                }
            )

        self.es.es_client.bulk(operations=operations, refresh=True)

    def ensure_index(self, index_name: str, dims: int) -> None:
        if self.es.es_client.indices.exists(index=index_name):
            return

        self.es.es_client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "name": {"type": "keyword"},
                            "entity_ids": {"type": "keyword"},
                            "relation_ids": {"type": "keyword"},
                            "passage_ids": {"type": "keyword"},
                            "subject": {"type": "keyword"},
                            "predicate": {"type": "keyword"},
                            "object": {"type": "keyword"},
                        }
                    },
                }
            },
        )

    def get_chat_model(self):
        if self.chat_model is None:
            self.chat_model = get_chat_model()
        return self.chat_model

    @classmethod
    def parse_triplets(cls, raw_triplets: Any) -> List[Tuple[str, str, str]]:
        triplets = []
        for item in raw_triplets or []:
            if isinstance(item, dict):
                subject = item.get("subject") or item.get("head")
                predicate = item.get("predicate") or item.get("relation")
                object_ = item.get("object") or item.get("tail")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                subject, predicate, object_ = item[:3]
            else:
                continue

            subject = str(subject or "").strip()
            predicate = str(predicate or "").strip()
            object_ = str(object_ or "").strip()
            if subject and predicate and object_:
                triplets.append((subject, predicate, object_))

        return list(dict.fromkeys(triplets))

    @classmethod
    def get_entity_id(cls, entity_name: str, entity_name_to_id: Dict[str, str]) -> str:
        normalized = cls.normalize(entity_name)
        if normalized not in entity_name_to_id:
            entity_name_to_id[normalized] = cls.stable_id("ent", normalized)
        return entity_name_to_id[normalized]

    @classmethod
    def stable_id(cls, prefix: str, text: str) -> str:
        digest = hashlib.md5(cls.normalize(text).encode("utf-8")).hexdigest()
        return f"{prefix}_{digest}"

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(str(text).lower().strip().split())

    @staticmethod
    def load_json_object(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def simple_extract_entities(query: str) -> List[str]:
        words = []
        for raw_word in query.replace("，", " ").replace("。", " ").split():
            word = raw_word.strip("'\".,;:!?()[]{}<>《》、")
            if len(word) >= 2:
                words.append(word)
        return list(dict.fromkeys(words))[:8]


def graph_index_names(prefix: str) -> Dict[str, str]:
    return ElasticGraphRAG.index_names(prefix)


def add_texts_to_elastic_graph(
    es: Elasticsearch,
    texts: List[str],
    graph_name: str,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    extract_triplets: bool = True,
) -> Dict[str, Any]:
    rag = ElasticGraphRAG(es=es, graph_name=graph_name)
    return rag.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        extract_triplets=extract_triplets,
    )


def add_documents_to_elastic_graph(
    es: Elasticsearch,
    documents: List[Document],
    graph_name: str,
    extract_triplets: bool = True,
) -> Dict[str, Any]:
    rag = ElasticGraphRAG(es=es, graph_name=graph_name)
    return rag.add_documents(documents=documents, extract_triplets=extract_triplets)


def retrieve_from_elastic_graph(
    es: Elasticsearch,
    query: str,
    graph_name: str,
    k: int = 6,
    entity_top_k: int = 5,
    relation_top_k: int = 8,
    expansion_degree: int = 1,
    relation_limit: int = 30,
    return_debug: bool = False,
) -> List[Dict[str, Any]] | Dict[str, Any]:
    rag = ElasticGraphRAG(es=es, graph_name=graph_name)
    return rag.retrieve(
        query=query,
        k=k,
        entity_top_k=entity_top_k,
        relation_top_k=relation_top_k,
        expansion_degree=expansion_degree,
        relation_limit=relation_limit,
        return_debug=return_debug,
    )


def extract_triplets_from_text(text: str) -> List[Tuple[str, str, str]]:
    return ElasticGraphRAG(es=None, graph_name="default").extract_triplets(text)


def extract_query_entities(query: str) -> List[str]:
    return ElasticGraphRAG(es=None, graph_name="default").extract_query_entities(query)
