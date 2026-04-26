from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from elasticsearch import NotFoundError
from langchain_core.documents import Document
from loguru import logger
from pydantic import BaseModel, Field

from langchain_api.constant import workspace_path
from langchain_api.rag.elastic_graph_rag import ElasticGraphRAG
from langchain_api.rag.elastic_utils import Elasticsearch
from langchain_api.rag.text_splitter import PDFParser
from langchain_api.settings import settings
from langchain_api.utils import get_embedding_model


class KnowledgeBaseRecord(BaseModel):
    knowledge_base_id: str
    user_id: str
    name: str
    description: str = ""
    index_prefix: str
    passage_index: str
    entity_index: str
    relation_index: str
    document_count: int = 0
    chunk_count: int = 0
    created_at: str
    updated_at: str


class KnowledgeBaseDocumentRecord(BaseModel):
    document_id: str
    knowledge_base_id: str
    user_id: str
    file_name: str
    display_name: str
    content_type: str = ""
    file_size: int = 0
    chunk_count: int = 0
    storage_path: str
    created_at: str
    updated_at: str


class KnowledgeBaseDeleteResult(BaseModel):
    knowledge_base: KnowledgeBaseRecord
    deleted_documents: int
    deleted_indexes: dict[str, str]


class KnowledgeBaseUploadError(BaseModel):
    file_name: str
    error: str


class KnowledgeBaseUploadResponse(BaseModel):
    knowledge_base: KnowledgeBaseRecord
    documents: list[KnowledgeBaseDocumentRecord] = Field(default_factory=list)
    errors: list[KnowledgeBaseUploadError] = Field(default_factory=list)


class PaginatedKnowledgeBaseResponse(BaseModel):
    items: list[KnowledgeBaseRecord]
    total: int
    page: int
    page_size: int


class PaginatedKnowledgeBaseDocumentResponse(BaseModel):
    items: list[KnowledgeBaseDocumentRecord]
    total: int
    page: int
    page_size: int


class BulkDeleteKnowledgeBaseResponse(BaseModel):
    deleted_ids: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)


class BulkDeleteDocumentResponse(BaseModel):
    deleted_ids: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    knowledge_base: KnowledgeBaseRecord | None = None


@dataclass(slots=True)
class UploadedKnowledgeFile:
    file_name: str
    content_type: str
    data: bytes


class KnowledgeBaseManager:
    KNOWLEDGE_BASE_INDEX = "rag_knowledge_bases"
    DOCUMENT_INDEX = "rag_knowledge_base_documents"
    STORAGE_ROOT = workspace_path / "pdf_files" / "knowledge_bases"

    def __init__(self, es: Elasticsearch):
        self.es = es
        self._ensure_metadata_indexes()

    def list_knowledge_bases(self, user_id: str) -> list[KnowledgeBaseRecord]:
        hits = self._search(
            index_name=self.KNOWLEDGE_BASE_INDEX,
            query={"bool": {"filter": [{"term": {"user_id": user_id}}]}},
            size=500,
            sort=[{"updated_at": {"order": "desc"}}],
        )
        return [KnowledgeBaseRecord(**hit["_source"]) for hit in hits]

    def search_knowledge_bases(
        self,
        user_id: str,
        *,
        search: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> PaginatedKnowledgeBaseResponse:
        page, page_size = self._normalize_page(page, page_size)
        hits, total = self._search_with_total(
            index_name=self.KNOWLEDGE_BASE_INDEX,
            query=self._build_query(
                filters=[{"term": {"user_id": user_id}}],
                search=search,
                fields=["name^3", "description"],
            ),
            size=page_size,
            from_=(page - 1) * page_size,
            sort=[{"updated_at": {"order": "desc"}}],
        )
        return PaginatedKnowledgeBaseResponse(
            items=[KnowledgeBaseRecord(**hit["_source"]) for hit in hits],
            total=total,
            page=page,
            page_size=page_size,
        )

    def create_knowledge_base(
        self, user_id: str, name: str, description: str = ""
    ) -> KnowledgeBaseRecord:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Knowledge base name is required.")

        knowledge_base_id = uuid.uuid4().hex
        index_prefix = f"kb_{knowledge_base_id}"
        indexes = ElasticGraphRAG.index_names(index_prefix)
        now = self._now()
        source = {
            "knowledge_base_id": knowledge_base_id,
            "user_id": user_id,
            "name": normalized_name,
            "description": description.strip(),
            "index_prefix": index_prefix,
            "passage_index": indexes["passage"],
            "entity_index": indexes["entity"],
            "relation_index": indexes["relation"],
            "document_count": 0,
            "chunk_count": 0,
            "created_at": now,
            "updated_at": now,
        }
        self.es.es_client.index(
            index=self.KNOWLEDGE_BASE_INDEX,
            id=knowledge_base_id,
            document=source,
            refresh=True,
        )
        return KnowledgeBaseRecord(**source)

    def get_knowledge_base(self, user_id: str, knowledge_base_id: str) -> KnowledgeBaseRecord:
        source = self._get_owned_document(
            index_name=self.KNOWLEDGE_BASE_INDEX,
            document_id=knowledge_base_id,
            user_id=user_id,
            error_message="Knowledge base not found.",
        )
        return KnowledgeBaseRecord(**source)

    def update_knowledge_base(
        self,
        user_id: str,
        knowledge_base_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBaseRecord:
        source = self._get_owned_document(
            index_name=self.KNOWLEDGE_BASE_INDEX,
            document_id=knowledge_base_id,
            user_id=user_id,
            error_message="Knowledge base not found.",
        )

        if name is not None:
            normalized_name = name.strip()
            if not normalized_name:
                raise ValueError("Knowledge base name is required.")
            source["name"] = normalized_name
        if description is not None:
            source["description"] = description.strip()
        source["updated_at"] = self._now()

        self.es.es_client.index(
            index=self.KNOWLEDGE_BASE_INDEX,
            id=knowledge_base_id,
            document=source,
            refresh=True,
        )
        return KnowledgeBaseRecord(**source)

    def delete_knowledge_base(
        self, user_id: str, knowledge_base_id: str
    ) -> KnowledgeBaseDeleteResult:
        knowledge_base = self.get_knowledge_base(user_id, knowledge_base_id)
        rag = ElasticGraphRAG(self.es, knowledge_base.index_prefix)
        graph_result = rag.delete_graph(ignore_missing=True)

        document_ids = [
            item.document_id
            for item in self.list_documents(user_id, knowledge_base_id)
        ]
        if document_ids:
            self.es.es_client.bulk(
                operations=[
                    {"delete": {"_index": self.DOCUMENT_INDEX, "_id": document_id}}
                    for document_id in document_ids
                ],
                refresh=True,
            )

        self.es.es_client.delete(
            index=self.KNOWLEDGE_BASE_INDEX,
            id=knowledge_base_id,
            refresh=True,
        )

        return KnowledgeBaseDeleteResult(
            knowledge_base=knowledge_base,
            deleted_documents=len(document_ids),
            deleted_indexes=graph_result["result"],
        )

    def list_documents(
        self, user_id: str, knowledge_base_id: str
    ) -> list[KnowledgeBaseDocumentRecord]:
        self.get_knowledge_base(user_id, knowledge_base_id)
        hits = self._search(
            index_name=self.DOCUMENT_INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"user_id": user_id}},
                        {"term": {"knowledge_base_id": knowledge_base_id}},
                    ]
                }
            },
            size=1000,
            sort=[{"created_at": {"order": "desc"}}],
        )
        return [KnowledgeBaseDocumentRecord(**hit["_source"]) for hit in hits]

    def search_documents(
        self,
        user_id: str,
        knowledge_base_id: str,
        *,
        search: str = "",
        page: int = 1,
        page_size: int = 10,
    ) -> PaginatedKnowledgeBaseDocumentResponse:
        self.get_knowledge_base(user_id, knowledge_base_id)
        page, page_size = self._normalize_page(page, page_size)
        hits, total = self._search_with_total(
            index_name=self.DOCUMENT_INDEX,
            query=self._build_query(
                filters=[
                    {"term": {"user_id": user_id}},
                    {"term": {"knowledge_base_id": knowledge_base_id}},
                ],
                search=search,
                fields=["display_name^3", "file_name"],
            ),
            size=page_size,
            from_=(page - 1) * page_size,
            sort=[{"created_at": {"order": "desc"}}],
        )
        return PaginatedKnowledgeBaseDocumentResponse(
            items=[KnowledgeBaseDocumentRecord(**hit["_source"]) for hit in hits],
            total=total,
            page=page,
            page_size=page_size,
        )

    def update_document(
        self,
        user_id: str,
        knowledge_base_id: str,
        document_id: str,
        *,
        display_name: str,
    ) -> KnowledgeBaseDocumentRecord:
        self.get_knowledge_base(user_id, knowledge_base_id)
        source = self._get_owned_document(
            index_name=self.DOCUMENT_INDEX,
            document_id=document_id,
            user_id=user_id,
            error_message="Document not found.",
        )
        if source["knowledge_base_id"] != knowledge_base_id:
            raise ValueError("Document does not belong to this knowledge base.")

        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("Document name is required.")

        source["display_name"] = normalized_name
        source["updated_at"] = self._now()
        self.es.es_client.index(
            index=self.DOCUMENT_INDEX,
            id=document_id,
            document=source,
            refresh=True,
        )
        return KnowledgeBaseDocumentRecord(**source)

    def delete_document(
        self, user_id: str, knowledge_base_id: str, document_id: str
    ) -> dict[str, Any]:
        knowledge_base = self.get_knowledge_base(user_id, knowledge_base_id)
        source = self._get_owned_document(
            index_name=self.DOCUMENT_INDEX,
            document_id=document_id,
            user_id=user_id,
            error_message="Document not found.",
        )
        if source["knowledge_base_id"] != knowledge_base_id:
            raise ValueError("Document does not belong to this knowledge base.")

        passage_ids = self._search_ids_by_term(
            index_name=knowledge_base.passage_index,
            field="metadata.file_id",
            value=document_id,
            size=10000,
        )
        rag = ElasticGraphRAG(self.es, knowledge_base.index_prefix)
        delete_result = rag.delete_documents(passage_ids)

        self.es.es_client.delete(
            index=self.DOCUMENT_INDEX,
            id=document_id,
            refresh=True,
        )
        knowledge_base = self._refresh_knowledge_base_stats(knowledge_base)

        return {
            "knowledge_base": knowledge_base,
            "document_id": document_id,
            "deleted_passages": delete_result["deleted_passages"],
            "deleted_relations": delete_result["deleted_relations"],
            "deleted_entities": delete_result["deleted_entities"],
        }

    def bulk_delete_knowledge_bases(
        self, user_id: str, knowledge_base_ids: list[str]
    ) -> BulkDeleteKnowledgeBaseResponse:
        deleted_ids: list[str] = []
        failed: dict[str, str] = {}
        for knowledge_base_id in knowledge_base_ids:
            try:
                self.delete_knowledge_base(user_id, knowledge_base_id)
                deleted_ids.append(knowledge_base_id)
            except Exception as exc:  # noqa: BLE001
                failed[knowledge_base_id] = str(exc)
        return BulkDeleteKnowledgeBaseResponse(
            deleted_ids=deleted_ids,
            failed=failed,
        )

    def bulk_delete_documents(
        self,
        user_id: str,
        knowledge_base_id: str,
        document_ids: list[str],
    ) -> BulkDeleteDocumentResponse:
        deleted_ids: list[str] = []
        failed: dict[str, str] = {}
        for document_id in document_ids:
            try:
                self.delete_document(user_id, knowledge_base_id, document_id)
                deleted_ids.append(document_id)
            except Exception as exc:  # noqa: BLE001
                failed[document_id] = str(exc)

        knowledge_base: KnowledgeBaseRecord | None = None
        try:
            knowledge_base = self.get_knowledge_base(user_id, knowledge_base_id)
        except Exception:  # noqa: BLE001
            knowledge_base = None

        return BulkDeleteDocumentResponse(
            deleted_ids=deleted_ids,
            failed=failed,
            knowledge_base=knowledge_base,
        )

    def upload_documents(
        self,
        user_id: str,
        knowledge_base_id: str,
        files: Iterable[UploadedKnowledgeFile],
    ) -> KnowledgeBaseUploadResponse:
        knowledge_base = self.get_knowledge_base(user_id, knowledge_base_id)
        rag = ElasticGraphRAG(self.es, knowledge_base.index_prefix)
        storage_dir = self._storage_dir(user_id, knowledge_base_id)
        storage_dir.mkdir(parents=True, exist_ok=True)

        documents: list[KnowledgeBaseDocumentRecord] = []
        errors: list[KnowledgeBaseUploadError] = []

        for uploaded_file in files:
            try:
                document_record = self._ingest_file(
                    user_id=user_id,
                    knowledge_base=knowledge_base,
                    rag=rag,
                    storage_dir=storage_dir,
                    uploaded_file=uploaded_file,
                )
                documents.append(document_record)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Knowledge file upload failed: {}", uploaded_file.file_name)
                errors.append(
                    KnowledgeBaseUploadError(
                        file_name=uploaded_file.file_name,
                        error=str(exc),
                    )
                )

        knowledge_base = self._refresh_knowledge_base_stats(knowledge_base)
        return KnowledgeBaseUploadResponse(
            knowledge_base=knowledge_base,
            documents=documents,
            errors=errors,
        )

    def _ingest_file(
        self,
        *,
        user_id: str,
        knowledge_base: KnowledgeBaseRecord,
        rag: ElasticGraphRAG,
        storage_dir: Path,
        uploaded_file: UploadedKnowledgeFile,
    ) -> KnowledgeBaseDocumentRecord:
        original_file_name = Path(uploaded_file.file_name or "unnamed").name
        if not original_file_name:
            raise ValueError("Uploaded file name is required.")

        document_id = uuid.uuid4().hex
        storage_name = f"{document_id}_{self._safe_file_name(original_file_name)}"
        storage_path = storage_dir / storage_name
        storage_path.write_bytes(uploaded_file.data)

        parser = PDFParser(
            bucket_name=self._storage_bucket_name(
                user_id=user_id,
                knowledge_base_id=knowledge_base.knowledge_base_id,
            ),
            file_path=storage_name,
            file_id=document_id,
        )
        chunks = parser.get_chunk()
        prepared_documents = self._prepare_documents(
            knowledge_base=knowledge_base,
            user_id=user_id,
            document_id=document_id,
            storage_name=storage_name,
            storage_path=storage_path,
            original_file_name=original_file_name,
            content_type=uploaded_file.content_type,
            chunks=chunks,
        )

        rag.add_documents(prepared_documents, extract_triplets=True)

        now = self._now()
        source = {
            "document_id": document_id,
            "knowledge_base_id": knowledge_base.knowledge_base_id,
            "user_id": user_id,
            "file_name": original_file_name,
            "display_name": original_file_name,
            "content_type": uploaded_file.content_type or "",
            "file_size": len(uploaded_file.data),
            "chunk_count": len(prepared_documents),
            "storage_path": str(storage_path),
            "created_at": now,
            "updated_at": now,
        }
        self.es.es_client.index(
            index=self.DOCUMENT_INDEX,
            id=document_id,
            document=source,
            refresh=True,
        )
        return KnowledgeBaseDocumentRecord(**source)

    def _prepare_documents(
        self,
        *,
        knowledge_base: KnowledgeBaseRecord,
        user_id: str,
        document_id: str,
        storage_name: str,
        storage_path: Path,
        original_file_name: str,
        content_type: str,
        chunks: list[Document],
    ) -> list[Document]:
        prepared_documents: list[Document] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = dict(chunk.metadata or {})
            segment_id = metadata.get("segment_id") or index
            metadata.update(
                {
                    "user_id": user_id,
                    "knowledge_base_id": knowledge_base.knowledge_base_id,
                    "knowledge_base_name": knowledge_base.name,
                    "document_id": document_id,
                    "file_name": original_file_name,
                    "display_name": original_file_name,
                    "storage_name": storage_name,
                    "storage_path": str(storage_path),
                    "content_type": content_type or "",
                }
            )
            prepared_documents.append(
                Document(
                    id=f"{document_id}_{segment_id}",
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )
        return prepared_documents

    def _refresh_knowledge_base_stats(
        self, knowledge_base: KnowledgeBaseRecord
    ) -> KnowledgeBaseRecord:
        source = self._get_owned_document(
            index_name=self.KNOWLEDGE_BASE_INDEX,
            document_id=knowledge_base.knowledge_base_id,
            user_id=knowledge_base.user_id,
            error_message="Knowledge base not found.",
        )
        source["document_count"] = self._count_documents(
            index_name=self.DOCUMENT_INDEX,
            query={
                "bool": {
                    "filter": [
                        {"term": {"user_id": knowledge_base.user_id}},
                        {"term": {"knowledge_base_id": knowledge_base.knowledge_base_id}},
                    ]
                }
            },
        )
        source["chunk_count"] = self._count_index(source["passage_index"])
        source["updated_at"] = self._now()
        self.es.es_client.index(
            index=self.KNOWLEDGE_BASE_INDEX,
            id=knowledge_base.knowledge_base_id,
            document=source,
            refresh=True,
        )
        return KnowledgeBaseRecord(**source)

    def _ensure_metadata_indexes(self) -> None:
        if not self.es.es_client.indices.exists(index=self.KNOWLEDGE_BASE_INDEX):
            self.es.es_client.indices.create(
                index=self.KNOWLEDGE_BASE_INDEX,
                mappings={
                    "properties": {
                        "knowledge_base_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        },
                        "description": {"type": "text"},
                        "index_prefix": {"type": "keyword"},
                        "passage_index": {"type": "keyword"},
                        "entity_index": {"type": "keyword"},
                        "relation_index": {"type": "keyword"},
                        "document_count": {"type": "integer"},
                        "chunk_count": {"type": "integer"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                    }
                },
            )

        if not self.es.es_client.indices.exists(index=self.DOCUMENT_INDEX):
            self.es.es_client.indices.create(
                index=self.DOCUMENT_INDEX,
                mappings={
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "knowledge_base_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "file_name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        },
                        "display_name": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        },
                        "content_type": {"type": "keyword"},
                        "file_size": {"type": "long"},
                        "chunk_count": {"type": "integer"},
                        "storage_path": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                    }
                },
            )

    def _get_owned_document(
        self,
        *,
        index_name: str,
        document_id: str,
        user_id: str,
        error_message: str,
    ) -> dict[str, Any]:
        try:
            result = self.es.es_client.get(index=index_name, id=document_id)
        except NotFoundError as exc:
            raise ValueError(error_message) from exc

        source = result["_source"]
        if source.get("user_id") != user_id:
            raise ValueError(error_message)
        return source

    def _search(
        self,
        *,
        index_name: str,
        query: dict[str, Any],
        size: int,
        sort: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.es.es_client.indices.exists(index=index_name):
            return []
        body: dict[str, Any] = {"query": query}
        if sort:
            body["sort"] = sort
        results = self.es.es_client.search(index=index_name, body=body, size=size)
        return results["hits"]["hits"]

    def _search_with_total(
        self,
        *,
        index_name: str,
        query: dict[str, Any],
        size: int,
        from_: int = 0,
        sort: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.es.es_client.indices.exists(index=index_name):
            return [], 0
        body: dict[str, Any] = {
            "query": query,
            "from": from_,
            "track_total_hits": True,
        }
        if sort:
            body["sort"] = sort
        results = self.es.es_client.search(index=index_name, body=body, size=size)
        total = int(results["hits"]["total"]["value"])
        return results["hits"]["hits"], total

    def _build_query(
        self,
        *,
        filters: list[dict[str, Any]],
        search: str,
        fields: list[str],
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"bool": {"filter": filters}}
        normalized_search = search.strip()
        if normalized_search:
            query["bool"]["must"] = [
                {
                    "multi_match": {
                        "query": normalized_search,
                        "fields": fields,
                        "type": "best_fields",
                    }
                }
            ]
        return query

    def _normalize_page(self, page: int, page_size: int) -> tuple[int, int]:
        normalized_page = max(1, int(page))
        normalized_page_size = max(1, min(100, int(page_size)))
        return normalized_page, normalized_page_size

    def _search_ids_by_term(
        self, *, index_name: str, field: str, value: str, size: int
    ) -> list[str]:
        if not self.es.es_client.indices.exists(index=index_name):
            return []
        results = self.es.es_client.search(
            index=index_name,
            body={"query": {"term": {field: value}}, "_source": False},
            size=size,
        )
        return [hit["_id"] for hit in results["hits"]["hits"]]

    def _count_documents(self, *, index_name: str, query: dict[str, Any]) -> int:
        if not self.es.es_client.indices.exists(index=index_name):
            return 0
        result = self.es.es_client.count(index=index_name, body={"query": query})
        return int(result["count"])

    def _count_index(self, index_name: str) -> int:
        if not self.es.es_client.indices.exists(index=index_name):
            return 0
        result = self.es.es_client.count(
            index=index_name,
            body={"query": {"match_all": {}}},
        )
        return int(result["count"])

    def _storage_dir(self, user_id: str, knowledge_base_id: str) -> Path:
        return self.STORAGE_ROOT / self._safe_path_part(user_id) / knowledge_base_id

    def _storage_bucket_name(self, user_id: str, knowledge_base_id: str) -> str:
        return (
            Path("knowledge_bases") / self._safe_path_part(user_id) / knowledge_base_id
        ).as_posix()

    @staticmethod
    def _safe_path_part(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
        return normalized.strip("_") or "default"

    @staticmethod
    def _safe_file_name(file_name: str) -> str:
        normalized = re.sub(r"[^\w.\- ]+", "_", file_name.strip())
        normalized = normalized.replace(" ", "_")
        return normalized or "unnamed"

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")


embeddings = get_embedding_model()
knowledge_base_manager = KnowledgeBaseManager(
    Elasticsearch(
        url=settings.ES_URL,
        username=settings.ES_URSR,
        password=settings.ES_PWD,
        embedding_model=embeddings,
    )
)
