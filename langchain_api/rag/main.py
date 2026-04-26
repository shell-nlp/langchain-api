from fastapi import APIRouter

from langchain_api.rag.service import add_rag_api_endpoint


def create_rag_router() -> APIRouter:
    router = APIRouter(prefix="/api/rag")
    add_rag_api_endpoint(app=router, path="/general_api")
    return router


__all__ = ["create_rag_router"]
