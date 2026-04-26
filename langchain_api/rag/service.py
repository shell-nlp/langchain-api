from fastapi import APIRouter, FastAPI
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver

from langchain_api.api import add_general_api_endpoint
from langchain_api.middleware import RAGMiddleware
from langchain_api.middleware.common import BusinessMiddleware
from langchain_api.rag.context import AgentContext
from langchain_api.rag.retriever import vector_store
from langchain_api.settings import settings


def create_rag_agent():
    model = ChatDeepSeek(
        model=settings.CHAT_MODEL_NAME,
        tags=["agent"],
        api_base=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
    )
    rewrite_model = ChatDeepSeek(
        model=settings.CHAT_MODEL_NAME,
        tags=["rag"],
        api_base=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
    )
    return create_agent(
        model=model,
        middleware=[
            RAGMiddleware(
                vector_store=vector_store,
                rewrite_query=True,
                model=rewrite_model,
                retrieve_router=True,
            ),
            BusinessMiddleware(),
        ],
        checkpointer=MemorySaver(),
    )


def add_rag_api_endpoint(
    app: FastAPI | APIRouter,
    path: str = "/api/rag/general_api",
):
    rag_agent = create_rag_agent()
    add_general_api_endpoint(
        app=app,
        agent=rag_agent,
        path=path,
        context=AgentContext,
        name="rag_general_api",
    )
    return rag_agent
