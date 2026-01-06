from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_api.middleware import RAGMiddleware
import os

os.system("clear")
model = ChatOpenAI(
    model="qwen3",
    base_url="http://localhost:8082/v1",
    api_key="your_api_key",
    tags=["rag"],
)

agent = create_agent(
    model=model,
    middleware=[
        RAGMiddleware(),
    ],
)


for mode, chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": """你好啊""",
            }
        ]
    },
    stream_mode=["messages", "updates"],
):
    if mode == "messages":  # 只处理消息流
        msg, metadata = chunk
        if metadata.get("tags", []) == ["rag"]:
            if msg.content:
                print(msg.content, end="", flush=True)
print()
