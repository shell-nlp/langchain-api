from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_api.middleware import RAGMiddleware
from langchain_api.retriever import vector_store
import os

os.system("clear")

model = ChatOpenAI(
    model="qwen3",
    base_url="http://localhost:8082/v1",
    api_key="your_api_key",
    tags=["rag"],
)
rewrite_model = ChatOpenAI(
    model="qwen3",
    base_url="http://localhost:8082/v1",
    api_key="your_api_key",
)

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
)


for mode, chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": """李四的职位是什么？""",
            }
        ]
    },
    stream_mode=["messages", "updates"],
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
