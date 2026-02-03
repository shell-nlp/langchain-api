from typing import Literal
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langchain_api.agent.agent import Agent
from langgraph.types import Command
from pydantic import BaseModel, Field
from loguru import logger
import uuid
import os
from pathlib import Path

root_path = Path(__file__).parent.parent.parent

frontend_path = root_path / "frontend"
os.system("clear")
agent = Agent().get_agent()
app = FastAPI()
# 将 html 路由到 /
app.mount(
    "/web",
    StaticFiles(
        directory=frontend_path,
        html=True,
    ),
    name="frontend",
)


@app.get("/")
def redirect_to_frontend():
    return RedirectResponse(url="/web/index.html")


class Request(BaseModel):
    query: str | None = Field(
        None,
        description="用户输入的查询",
        examples=[
            "请你执行如下任务：\n 1. 计算 10 + 10 的结果。\n2. 将结果乘以 5。\n 3. 根据结果生成一个故事。"
        ],
    )
    resume: dict | None = Field(
        None, description="恢复信息", examples=[{"decisions": [{"type": "approve"}]}]
    )
    # session_id 默认随机的uuid
    session_id: str = str(uuid.uuid4())


class StreamResponse(BaseModel):

    event: Literal["token", "tool_calls", "tool_output", "__interrupt__"] = "token"
    data: dict | None = None


@app.post("/agent_chat", response_model=StreamResponse)
def agent_chat(request: Request):
    logger.debug(f"request: \n{request.model_dump_json(indent=2)}")
    config = {"configurable": {"thread_id": f"{request.session_id}"}}

    update = None
    input = None

    if request.query and request.resume:
        raise ValueError("query 和 resume 不能同时存在")
    elif request.query:
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""{request.query}""",
                }
            ]
        }
        input = update
    elif request.resume:
        input = Command(resume=request.resume)

    stream_response = StreamResponse()

    def stream_generator():
        text = ""
        for mode, chunk in agent.stream(
            input=input,
            stream_mode=["messages", "updates"],
            config=config,
        ):
            if mode == "messages":  # 只处理消息流
                msg, metadata = chunk
                if metadata.get("tags", []) == ["agent"]:
                    if msg.content:
                        text += msg.content
                        stream_response.event = "token"
                        stream_response.data = {"token": msg.content, "id": msg.id}
                        yield f"data: {stream_response.model_dump_json()}\n\n"
            elif mode == "updates":  # 处理更新流
                # print(f"\n[Update]: {chunk}")
                if "__interrupt__" in chunk:  # 处理 Human in the Loop
                    stream_response.event = "__interrupt__"
                    stream_response.data = {
                        "__interrupt__": chunk["__interrupt__"][0].value
                    }
                    yield f"data: {stream_response.model_dump_json()}\n\n"

                if "model" in chunk and not chunk["model"]["messages"][0].tool_calls:
                    # 这个是最后一次整体的 tool 内容
                    ...
                if "model" in chunk and chunk["model"]["messages"][0].tool_calls:
                    stream_response.event = "tool_calls"
                    stream_response.data = {
                        "tool_calls": chunk["model"]["messages"][0].tool_calls,
                        "id": chunk["model"]["messages"][0].id,
                    }
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                    text += f"\n{'-'*100}\n"

                if "tools" in chunk:
                    stream_response.event = "tool_output"
                    stream_response.data = {
                        "tool_output": chunk["tools"]["messages"],
                    }
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                    text += f"\n工具响应： \n{chunk['tools']['messages']}\n{'-'*100}\n"

        logger.info(f"session_id：{request.session_id} \nFinal Response: \n{text}")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7869)
