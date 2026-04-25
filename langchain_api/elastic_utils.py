from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch as ESClient
from loguru import logger

from langchain_api.utils import get_embedding_model


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
