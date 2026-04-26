from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver

from langchain_api.api import add_general_api_endpoint
from langchain_api.middleware import RAGMiddleware
from langchain_api.rag.context import AgentContext
from langchain_api.rag.retriever import vector_store
from langchain_api.settings import settings
from langchain_api.middleware.common import BusinessMiddleware

app = FastAPI()

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
checkpointer = MemorySaver()
agent = create_agent(
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
    checkpointer=checkpointer,
)

# 添加 general_api 端点
add_general_api_endpoint(
    app=app,
    agent=agent,
    path="/api/general_api",
    context=AgentContext,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7869)
