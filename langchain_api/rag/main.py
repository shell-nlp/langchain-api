from fastapi import APIRouter

from langchain_api.rag.management_api import add_knowledge_base_management_endpoints
from langchain_api.rag.service import add_rag_api_endpoint


def create_rag_router() -> APIRouter:
    router = APIRouter(prefix="/api/rag")
    add_knowledge_base_management_endpoints(router)
    add_rag_api_endpoint(app=router, path="/general_api")
    return router


__all__ = ["create_rag_router"]
