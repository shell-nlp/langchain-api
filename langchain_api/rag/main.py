from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_api.middleware import RAGMiddleware
from langchain_api.retriever import vector_store
from langgraph.checkpoint.memory import MemorySaver
from langchain_api.settings import settings
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
    query = input("问题：")
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
        if mode == "messages":  # 只处理消息流
            msg, metadata = chunk
            if metadata.get("tags", []) == ["rag"]:
                if msg.content:
                    print(msg.content, end="", flush=True)
                    # print(msg.content)
        elif mode == "updates":  # 处理更新流
            update = chunk
            print(f"\n[Update]: {update}")
            # if "model" in update and not update["model"]["messages"][0].tool_calls:
            #     # print("ai: ", update["model"]["messages"][0].content, "\n", "-" * 50)
            #     pass
            # if "model" in update and update["model"]["messages"][0].tool_calls:
            #     print(
            #         "ai: ",
            #         f"调用工具信息：{update["model"]["messages"][0].tool_calls}",
            #         "\n",
            #         "-" * 50,
            #     )
            # if "tools" in update:

            #     print(
            #         "tools: ",
            #         f"调用工具：{update["tools"]["messages"][0].name} 结果： \n",
            #         update["tools"]["messages"][0].content,
            #         "\n",
            #         "-" * 50,
            #     )
    print()
