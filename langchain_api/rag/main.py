from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_api.middleware import RAGMiddleware
from langchain_api.retriever import vector_store
from langgraph.checkpoint.memory import MemorySaver
from langchain_api.settting import settings
import os

os.system("clear")

model = ChatOpenAI(
    model=settings.CHAT_MODEL_NAME,
    tags=["rag"],
)
rewrite_model = ChatOpenAI(
    model=settings.CHAT_MODEL_NAME,
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
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "123"}}
while True:
    query = input("请输入问题：")
    if query == "exit":
        break
    for mode, chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""{query}""",
                }
            ]
        },
        stream_mode=["messages", "updates"],
        config=config,
    ):
        if mode == "values":  # 处理最终值
            final_response = chunk
            print("\n[Final Response]:", final_response, flush=True)
        elif mode == "messages":  # 只处理消息流
            msg, metadata = chunk
            if metadata.get("tags", []) == ["rag"]:
                if msg.content:
                    print(msg.content, end="", flush=True)
                    # print(msg.content)
        elif mode == "updates":  # 处理更新流
            update = chunk
            print(f"\n[Update]: {update}", flush=True)
    print()
