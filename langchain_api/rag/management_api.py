from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from langchain_api.rag.knowledge_base import (
    BulkDeleteDocumentResponse,
    BulkDeleteKnowledgeBaseResponse,
    KnowledgeBaseDeleteResult,
    KnowledgeBaseDocumentDetailResponse,
    KnowledgeBaseDocumentRecord,
    KnowledgeBaseRecord,
    PaginatedKnowledgeBaseDocumentResponse,
    PaginatedKnowledgeBaseResponse,
    KnowledgeBaseUploadResponse,
    UploadedKnowledgeFile,
    knowledge_base_manager,
)


class KnowledgeBaseIdentityRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_id: str = Field(..., description="Knowledge base ID")


class KnowledgeBaseListRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    search: str = Field("", description="Search text")
    page: int = Field(1, description="Page number")
    page_size: int = Field(10, description="Page size")


class CreateKnowledgeBaseRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Knowledge base name")
    description: str = Field("", description="Knowledge base description")


class UpdateKnowledgeBaseRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_id: str = Field(..., description="Knowledge base ID")
    name: str | None = Field(None, description="Knowledge base name")
    description: str | None = Field(None, description="Knowledge base description")


class UpdateKnowledgeBaseDocumentRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_id: str = Field(..., description="Knowledge base ID")
    document_id: str = Field(..., description="Document ID")
    display_name: str = Field(..., description="Document display name")


class DeleteKnowledgeBaseDocumentRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_id: str = Field(..., description="Knowledge base ID")
    document_id: str = Field(..., description="Document ID")


class DocumentListRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_id: str = Field(..., description="Knowledge base ID")
    search: str = Field("", description="Search text")
    page: int = Field(1, description="Page number")
    page_size: int = Field(10, description="Page size")


class DocumentDetailRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_id: str = Field(..., description="Knowledge base ID")
    document_id: str = Field(..., description="Document ID")
    page: int = Field(1, description="Page number")
    page_size: int = Field(10, description="Page size")


class BulkDeleteKnowledgeBaseRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_ids: list[str] = Field(default_factory=list)


class BulkDeleteKnowledgeBaseDocumentRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    knowledge_base_id: str = Field(..., description="Knowledge base ID")
    document_ids: list[str] = Field(default_factory=list)


def _handle_value_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def add_knowledge_base_management_endpoints(router: APIRouter) -> None:
    @router.post("/knowledge-bases/list", response_model=PaginatedKnowledgeBaseResponse)
    def list_knowledge_bases(request: KnowledgeBaseListRequest):
        return knowledge_base_manager.search_knowledge_bases(
            user_id=request.user_id,
            search=request.search,
            page=request.page,
            page_size=request.page_size,
        )

    @router.post("/knowledge-bases/create", response_model=KnowledgeBaseRecord)
    def create_knowledge_base(request: CreateKnowledgeBaseRequest):
        try:
            return knowledge_base_manager.create_knowledge_base(
                user_id=request.user_id,
                name=request.name,
                description=request.description,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post("/knowledge-bases/detail", response_model=KnowledgeBaseRecord)
    def get_knowledge_base(request: KnowledgeBaseIdentityRequest):
        try:
            return knowledge_base_manager.get_knowledge_base(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post("/knowledge-bases/update", response_model=KnowledgeBaseRecord)
    def update_knowledge_base(request: UpdateKnowledgeBaseRequest):
        try:
            return knowledge_base_manager.update_knowledge_base(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
                name=request.name,
                description=request.description,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post("/knowledge-bases/delete", response_model=KnowledgeBaseDeleteResult)
    def delete_knowledge_base(request: KnowledgeBaseIdentityRequest):
        try:
            return knowledge_base_manager.delete_knowledge_base(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post(
        "/knowledge-bases/bulk-delete",
        response_model=BulkDeleteKnowledgeBaseResponse,
    )
    def bulk_delete_knowledge_bases(request: BulkDeleteKnowledgeBaseRequest):
        try:
            return knowledge_base_manager.bulk_delete_knowledge_bases(
                user_id=request.user_id,
                knowledge_base_ids=request.knowledge_base_ids,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post(
        "/knowledge-bases/documents/list",
        response_model=PaginatedKnowledgeBaseDocumentResponse,
    )
    def list_documents(request: DocumentListRequest):
        try:
            return knowledge_base_manager.search_documents(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
                search=request.search,
                page=request.page,
                page_size=request.page_size,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post(
        "/knowledge-bases/documents/detail",
        response_model=KnowledgeBaseDocumentDetailResponse,
    )
    def get_document_detail(request: DocumentDetailRequest):
        try:
            return knowledge_base_manager.get_document_detail(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
                document_id=request.document_id,
                page=request.page,
                page_size=request.page_size,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post(
        "/knowledge-bases/documents/upload",
        response_model=KnowledgeBaseUploadResponse,
    )
    async def upload_documents(
        user_id: str = Form(..., description="User ID"),
        knowledge_base_id: str = Form(..., description="Knowledge base ID"),
        files: list[UploadFile] = File(..., description="Uploaded files"),
    ):
        uploaded_files: list[UploadedKnowledgeFile] = []
        for file in files:
            uploaded_files.append(
                UploadedKnowledgeFile(
                    file_name=file.filename or "unnamed",
                    content_type=file.content_type or "",
                    data=await file.read(),
                )
            )
            await file.close()

        try:
            return knowledge_base_manager.upload_documents(
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                files=uploaded_files,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post(
        "/knowledge-bases/documents/update",
        response_model=KnowledgeBaseDocumentRecord,
    )
    def update_document(request: UpdateKnowledgeBaseDocumentRequest):
        try:
            return knowledge_base_manager.update_document(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
                document_id=request.document_id,
                display_name=request.display_name,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post("/knowledge-bases/documents/delete")
    def delete_document(request: DeleteKnowledgeBaseDocumentRequest):
        try:
            return knowledge_base_manager.delete_document(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
                document_id=request.document_id,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc

    @router.post(
        "/knowledge-bases/documents/bulk-delete",
        response_model=BulkDeleteDocumentResponse,
    )
    def bulk_delete_documents(request: BulkDeleteKnowledgeBaseDocumentRequest):
        try:
            return knowledge_base_manager.bulk_delete_documents(
                user_id=request.user_id,
                knowledge_base_id=request.knowledge_base_id,
                document_ids=request.document_ids,
            )
        except ValueError as exc:
            raise _handle_value_error(exc) from exc
