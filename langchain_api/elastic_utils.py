from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from elasticsearch import Elasticsearch as ESClient
from loguru import logger


class Elasticsearch:
    def __init__(
        self,
        url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        embedding_model=None,
    ):
        self._url = url
        self._username = username
        self._password = password
        self._embedding_model = embedding_model
        self._es_client: Optional[ESClient] = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from langchain_api.utils import get_embedding_model

            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def es_client(self) -> ESClient:
        if self._es_client is None:
            self._es_client = ESClient(
                hosts=[self._url],
                basic_auth=(self._username, self._password)
                if self._username and self._password
                else None,
            )
        return self._es_client

    def vector_search(
        self,
        query: str,
        k: int = 3,
        index_name: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for search operations")
        query_vector = self.embedding_model.embed_query(query)
        results = self.es_client.search(
            index=index_name,
            body={
                "query": {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_vector,
                        "num_candidates": k * 2,
                    }
                }
            },
            size=k,
        )
        processed_results = []
        for hit in results["hits"]["hits"]:
            score = hit["_score"]
            if min_similarity is not None and score < min_similarity:
                continue
            processed_results.append(
                {
                    "content": hit["_source"].get("content", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": score,
                }
            )
        return processed_results

    def keyword_search(
        self, query: str, k: int = 3, index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for search operations")
        results = self.es_client.search(
            index=index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "title", "summary"],
                        "type": "best_fields",
                        "boost": 0.3,
                    }
                }
            },
            size=k,
        )
        return [
            {
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
            }
            for hit in results["hits"]["hits"]
        ]

    def retrieve(
        self, query: str, k: int = 3, index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for retrieve operations")
        vector_results = self.vector_search(query, k, index_name)
        keyword_results = self.keyword_search(query, k, index_name)

        seen_contents = set()
        merged_results = []
        for doc in vector_results:
            content_hash = hash(doc.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged_results.append(doc)

        for doc in keyword_results:
            content_hash = hash(doc.get("content", ""))
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged_results.append(doc)

        logger.info(
            f"ES检索结果数量：向量检索{len(vector_results)}，关键字检索{len(keyword_results)}，合并后{len(merged_results)}"
        )
        return merged_results[:k]

    def vector_graph_retrieve(
        self,
        query: str,
        k: int = 6,
        index_name: Optional[str] = None,
        entity_index_name: Optional[str] = None,
        relation_index_name: Optional[str] = None,
        entity_top_k: int = 5,
        relation_top_k: int = 8,
        expansion_degree: int = 1,
        relation_limit: int = 30,
        min_similarity: Optional[float] = None,
        query_entities: Optional[List[str]] = None,
        return_debug: bool = False,
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """
        基于 Elasticsearch 的向量图 RAG 检索。

        借鉴 vector-graph-rag 的思想，但不引入图数据库：
        1. 用查询和查询实体做向量召回，找到种子关系/实体。
        2. 通过 ES 文档中的 metadata.entity_ids / metadata.relation_ids 做邻接扩展。
        3. 关系过多时，再用关系向量相似度做一次裁剪。
        4. 最后用保留下来的 relation_ids / entity_ids 找回原文片段。

        推荐索引结构：
        - passage index: content, embedding, metadata.entity_ids, metadata.relation_ids
        - entity index: content/name, embedding, metadata.relation_ids
        - relation index: content/text, embedding, metadata.entity_ids, metadata.passage_ids

        如果没有独立 entity/relation 索引，也可以只传 index_name，函数会退化为
        普通向量+关键词检索，同时保留相同返回格式。
        """
        if not index_name:
            raise ValueError("index_name is required for vector_graph_retrieve operations")

        entity_index_name = entity_index_name or index_name
        relation_index_name = relation_index_name or index_name
        query_entities = query_entities or self._simple_extract_entities(query)

        seed_entities = self._search_graph_items(
            texts=query_entities,
            index_name=entity_index_name,
            k=entity_top_k,
            min_similarity=min_similarity,
        )
        seed_relations = self._search_graph_items(
            texts=[query],
            index_name=relation_index_name,
            k=relation_top_k,
            min_similarity=min_similarity,
        )

        entity_ids = self._ids_from_hits(seed_entities)
        relation_ids = self._ids_from_hits(seed_relations)

        expanded_entity_ids, expanded_relation_ids, expansion_steps = self._expand_es_graph(
            entity_ids=entity_ids,
            relation_ids=relation_ids,
            entity_index_name=entity_index_name,
            relation_index_name=relation_index_name,
            degree=expansion_degree,
        )

        kept_relations, eviction = self._evict_relations_by_vector(
            query=query,
            relation_ids=expanded_relation_ids,
            relation_index_name=relation_index_name,
            limit=relation_limit,
        )

        passages = self._search_passages_by_graph(
            query=query,
            index_name=index_name,
            relation_ids=kept_relations,
            entity_ids=expanded_entity_ids,
            k=k,
        )

        if not passages:
            passages = self.retrieve(query=query, k=k, index_name=index_name)

        logger.info(
            "ES向量图RAG: query_entities={}, seed_entities={}, seed_relations={}, expanded_entities={}, expanded_relations={}, passages={}",
            len(query_entities),
            len(entity_ids),
            len(relation_ids),
            len(expanded_entity_ids),
            len(expanded_relation_ids),
            len(passages),
        )

        if not return_debug:
            return passages[:k]

        return {
            "query": query,
            "query_entities": query_entities,
            "passages": passages[:k],
            "seed_entity_ids": entity_ids,
            "seed_relation_ids": relation_ids,
            "expanded_entity_ids": expanded_entity_ids,
            "expanded_relation_ids": expanded_relation_ids,
            "kept_relation_ids": kept_relations,
            "eviction": eviction,
            "expansion_steps": expansion_steps,
        }

    def _search_graph_items(
        self,
        texts: List[str],
        index_name: str,
        k: int,
        min_similarity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for text in texts:
            if not text.strip():
                continue
            for item in self._vector_search_raw(text, k, index_name, min_similarity):
                item_id = item["id"]
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                hits.append(item)

        return hits

    def _vector_search_raw(
        self,
        query: str,
        k: int,
        index_name: str,
        min_similarity: Optional[float] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed_query(query)
        knn_query: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "num_candidates": max(k * 5, 20),
        }
        if ids:
            knn_query["filter"] = {
                "bool": {
                    "should": [
                        {"ids": {"values": ids}},
                        {"terms": {"metadata.id": ids}},
                    ],
                    "minimum_should_match": 1,
                }
            }

        results = self.es_client.search(
            index=index_name,
            body={"query": {"knn": knn_query}},
            size=k,
        )

        hits = []
        for hit in results["hits"]["hits"]:
            score = hit["_score"]
            if min_similarity is not None and score < min_similarity:
                continue
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            hits.append(
                {
                    "id": str(metadata.get("id") or hit["_id"]),
                    "es_id": hit["_id"],
                    "content": source.get("content") or source.get("text") or source.get("name") or "",
                    "metadata": metadata,
                    "score": score,
                }
            )
        return hits

    def _expand_es_graph(
        self,
        entity_ids: List[str],
        relation_ids: List[str],
        entity_index_name: str,
        relation_index_name: str,
        degree: int,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        all_entity_ids = set(entity_ids)
        all_relation_ids = set(relation_ids)
        steps: List[Dict[str, Any]] = []

        relation_ids_from_seed_entities = self._relations_by_entities(
            entity_ids=list(all_entity_ids),
            entity_index_name=entity_index_name,
            relation_index_name=relation_index_name,
        )
        new_relation_ids = relation_ids_from_seed_entities - all_relation_ids
        all_relation_ids.update(new_relation_ids)
        steps.append(
            {
                "step": 0,
                "operation": "entity_to_relation",
                "new_entity_ids": [],
                "new_relation_ids": sorted(new_relation_ids),
            }
        )

        for step in range(1, degree + 1):
            found_entity_ids = self._entities_by_relations(
                relation_ids=list(all_relation_ids), relation_index_name=relation_index_name
            )
            new_entity_ids = found_entity_ids - all_entity_ids
            all_entity_ids.update(new_entity_ids)

            found_relation_ids = self._relations_by_entities(
                entity_ids=list(new_entity_ids),
                entity_index_name=entity_index_name,
                relation_index_name=relation_index_name,
            )
            new_relation_ids = found_relation_ids - all_relation_ids
            all_relation_ids.update(new_relation_ids)

            steps.append(
                {
                    "step": step,
                    "operation": "relation_to_entity_to_relation",
                    "new_entity_ids": sorted(new_entity_ids),
                    "new_relation_ids": sorted(new_relation_ids),
                }
            )

            if not new_entity_ids and not new_relation_ids:
                break

        return sorted(all_entity_ids), sorted(all_relation_ids), steps

    def _relations_by_entities(
        self,
        entity_ids: List[str],
        entity_index_name: str,
        relation_index_name: str,
    ) -> Set[str]:
        relation_ids: Set[str] = set()
        for entity in self._get_docs_by_ids(entity_index_name, entity_ids):
            relation_ids.update(self._metadata_list(entity, "relation_ids"))

        if relation_ids:
            return relation_ids

        for relation in self._search_by_terms(
            index_name=relation_index_name,
            field="metadata.entity_ids",
            values=entity_ids,
            size=max(len(entity_ids) * 20, 50),
        ):
            relation_ids.add(relation["id"])
        return relation_ids

    def _entities_by_relations(
        self, relation_ids: List[str], relation_index_name: str
    ) -> Set[str]:
        entity_ids: Set[str] = set()
        for relation in self._get_docs_by_ids(relation_index_name, relation_ids):
            entity_ids.update(self._metadata_list(relation, "entity_ids"))
        return entity_ids

    def _evict_relations_by_vector(
        self,
        query: str,
        relation_ids: List[str],
        relation_index_name: str,
        limit: int,
    ) -> Tuple[List[str], Dict[str, Any]]:
        before_count = len(relation_ids)
        if before_count <= limit:
            return sorted(relation_ids), {
                "occurred": False,
                "before_count": before_count,
                "after_count": before_count,
            }

        kept = [
            hit["id"]
            for hit in self._vector_search_raw(
                query=query,
                k=limit,
                index_name=relation_index_name,
                ids=relation_ids,
            )
        ]
        return kept, {
            "occurred": True,
            "before_count": before_count,
            "after_count": len(kept),
        }

    def _search_passages_by_graph(
        self,
        query: str,
        index_name: str,
        relation_ids: List[str],
        entity_ids: List[str],
        k: int,
    ) -> List[Dict[str, Any]]:
        should_clauses = []
        if relation_ids:
            should_clauses.append({"terms": {"metadata.relation_ids": relation_ids}})
        if entity_ids:
            should_clauses.append({"terms": {"metadata.entity_ids": entity_ids}})
        if query:
            should_clauses.append(
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "title", "summary"],
                        "type": "best_fields",
                    }
                }
            )

        if not should_clauses:
            return []

        results = self.es_client.search(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1,
                    }
                }
            },
            size=k,
        )
        return [self._hit_to_result(hit) for hit in results["hits"]["hits"]]

    def _get_docs_by_ids(self, index_name: str, doc_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = [doc_id for doc_id in doc_ids if doc_id]
        if not ids:
            return []
        results = self.es_client.mget(index=index_name, ids=ids)
        docs = []
        found_ids = set()
        for doc in results.get("docs", []):
            if not doc.get("found"):
                continue
            source = doc.get("_source", {})
            metadata = source.get("metadata", {})
            found_ids.add(doc["_id"])
            if metadata.get("id") is not None:
                found_ids.add(str(metadata["id"]))
            docs.append(
                {
                    "id": doc["_id"],
                    "content": source.get("content") or source.get("text") or source.get("name") or "",
                    "metadata": metadata,
                }
            )

        missed_ids = [doc_id for doc_id in ids if doc_id not in found_ids]
        docs.extend(
            self._search_by_terms(
                index_name=index_name,
                field="metadata.id",
                values=missed_ids,
                size=len(missed_ids),
            )
        )
        return docs

    def _search_by_terms(
        self, index_name: str, field: str, values: List[str], size: int
    ) -> List[Dict[str, Any]]:
        if not values:
            return []
        results = self.es_client.search(
            index=index_name,
            body={"query": {"terms": {field: values}}},
            size=size,
        )
        return [self._hit_to_result(hit) for hit in results["hits"]["hits"]]

    def _hit_to_result(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        source = hit.get("_source", {})
        metadata = source.get("metadata", {})
        return {
            "id": str(metadata.get("id") or hit.get("_id")),
            "es_id": hit.get("_id"),
            "content": source.get("content") or source.get("text") or source.get("name") or "",
            "metadata": metadata,
            "score": hit.get("_score"),
        }

    def _ids_from_hits(self, hits: List[Dict[str, Any]]) -> List[str]:
        ids = []
        for hit in hits:
            ids.append(str(hit["id"]))
        return list(dict.fromkeys(ids))

    def _metadata_list(self, doc: Dict[str, Any], key: str) -> List[str]:
        value = doc.get("metadata", {}).get(key, [])
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _simple_extract_entities(self, query: str) -> List[str]:
        words = []
        for raw_word in query.replace("，", " ").replace("。", " ").split():
            word = raw_word.strip("'\".,;:!?()[]{}<>《》、")
            if len(word) >= 2:
                words.append(word)
        return list(dict.fromkeys(words))[:8]

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> str:
        if not index_name:
            raise ValueError("index_name is required for add operations")
        embedding = self.embedding_model.embed_query(content)
        doc_body = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        result = self.es_client.index(
            index=index_name,
            id=doc_id,
            document=doc_body,
            refresh=True,
        )
        logger.info(f"文档添加成功: id={result['_id']}, index={index_name}")
        return result["_id"]

    def add_batch(
        self,
        documents: List[Dict[str, Any]],
        index_name: Optional[str] = None,
    ) -> List[str]:
        if not index_name:
            raise ValueError("index_name is required for add_batch operations")
        if not documents:
            return []
        operations = []
        for doc in documents:
            content = doc.get("content", "")
            embedding = self.embedding_model.embed_query(content)
            operations.append({"index": {"_index": index_name}})
            operations.append(
                {
                    "content": content,
                    "embedding": embedding,
                    "metadata": doc.get("metadata", {}),
                }
            )

        result = self.es_client.bulk(operations=operations, refresh=True)
        ids = []
        for item in result["items"]:
            ids.append(item["index"]["_id"])
        logger.info(f"批量添加文档成功: 数量={len(ids)}, index={index_name}")
        return ids

    def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        if not index_name:
            raise ValueError("index_name is required for update operations")
        update_body: Dict[str, Any] = {}
        if content is not None:
            update_body["content"] = content
            update_body["embedding"] = self.embedding_model.embed_query(content)
        if metadata is not None:
            update_body["metadata"] = metadata

        if not update_body:
            logger.warning(f"文档更新失败: id={doc_id}, 无更新内容")
            return False

        result = self.es_client.update(
            index=index_name,
            id=doc_id,
            doc=update_body,
            refresh=True,
        )
        logger.info(f"文档更新成功: id={doc_id}, index={index_name}")
        return result["result"] in ["updated", "noop"]

    def delete(self, doc_id: str, index_name: Optional[str] = None) -> bool:
        if not index_name:
            raise ValueError("index_name is required for delete operations")
        result = self.es_client.delete(
            index=index_name,
            id=doc_id,
            refresh=True,
        )
        logger.info(f"文档删除成功: id={doc_id}, index={index_name}")
        return result["result"] == "deleted"

    def delete_batch(
        self, doc_ids: List[str], index_name: Optional[str] = None
    ) -> List[bool]:
        if not index_name:
            raise ValueError("index_name is required for delete_batch operations")
        if not doc_ids:
            return []

        operations = [
            {"delete": {"_index": index_name, "_id": doc_id}} for doc_id in doc_ids
        ]
        result = self.es_client.bulk(operations=operations, refresh=True)

        results = []
        for item in result["items"]:
            results.append(item["delete"]["result"] == "deleted")
        logger.info(
            f"批量删除文档成功: 成功数={sum(results)}, 失败数={len(results) - sum(results)}, index={index_name}"
        )
        return results

    def get(
        self, doc_id: str, index_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for get operations")
        try:
            result = self.es_client.get(index=index_name, id=doc_id)
            source = result["_source"]
            return {
                "content": source.get("content", ""),
                "metadata": source.get("metadata", {}),
            }
        except Exception as e:
            logger.error(f"获取文档失败: id={doc_id}, error={e}")
            return None

    def search(
        self,
        query: Optional[str] = None,
        k: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not index_name:
            raise ValueError("index_name is required for search operations")
        must_clauses = []
        if query:
            must_clauses.append(
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "title", "summary"],
                        "type": "best_fields",
                    }
                }
            )
        if filter_conditions:
            for field, value in filter_conditions.items():
                must_clauses.append({"term": {field: value}})

        search_body: Dict[str, Any] = (
            {"query": {"bool": {"must": must_clauses}}}
            if must_clauses
            else {"query": {"match_all": {}}}
        )

        results = self.es_client.search(index=index_name, body=search_body, size=k)
        return [
            {
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
            }
            for hit in results["hits"]["hits"]
        ]

    def exists(self, doc_id: str, index_name: Optional[str] = None) -> bool:
        if not index_name:
            raise ValueError("index_name is required for exists operations")
        return self.es_client.exists(index=index_name, id=doc_id)

    def count(
        self,
        filter_conditions: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> int:
        if not index_name:
            raise ValueError("index_name is required for count operations")
        if filter_conditions:
            must_clauses = [
                {"term": {field: value}} for field, value in filter_conditions.items()
            ]
            search_body = {"query": {"bool": {"must": must_clauses}}}
        else:
            search_body = {"query": {"match_all": {}}}
        result = self.es_client.count(index=index_name, body=search_body)
        return result["count"]
