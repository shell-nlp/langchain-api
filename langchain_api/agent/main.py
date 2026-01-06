from fastapi import FastAPI
from langchain_api.agent.agent import Agent
from langgraph.types import Command
import os

os.system("clear")
agent = Agent().get_agent()
app = FastAPI()

config = {"configurable": {"thread_id": "123"}}
for mode, chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": """请你执行如下任务：
    1. 计算 10 + 10 的结果。
    2. 将结果乘以 5。
    3. 根据结果生成一个故事。
    """,
            }
        ]
    },
    stream_mode=["messages", "updates"],
    config=config,
):
    if mode == "messages":  # 只处理消息流
        msg, metadata = chunk
        if metadata.get("tags", []) == ["agent"]:
            if msg.tool_calls or msg.content:
                # print(msg.content, end="", flush=True)
                print(msg.tool_calls)
    elif mode == "updates":  # 处理更新流
        update = chunk
        print(f"\n[Update]: {update}", flush=True)
# --------------------------------------------

for mode, chunk in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    stream_mode=["messages", "updates"],
    config=config,
):
    if mode == "messages":  # 只处理消息流
        msg, metadata = chunk
        if metadata.get("tags", []) == ["agent"]:
            if msg.tool_calls or msg.content:
                # print(msg.content, end="", flush=True)
                print(msg.tool_calls)
    elif mode == "updates":  # 处理更新流
        update = chunk
        print(f"\n[Update]: {update}", flush=True)
