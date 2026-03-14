from typing import Literal
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, Field


class GeneralAPIRequest(BaseModel):
    query: str | None = Field(
        None,
        description="用户输入的查询",
        examples=[
            "请你执行如下任务：\n 1. 计算 10 + 10 的结果。\n2. 将结果乘以 5。\n 3. 根据结果生成一个故事。"
        ],
    )
    resume: dict | None = Field(
        None,
        description="恢复信息",
        examples=[
            {"decisions": [{"type": "approve"}]},
            {
                "decisions": [
                    {
                        "type": "reject",
                        # 关于操作被拒绝原因的解释
                        "message": "不，这是错误的，因为......，而是这样做......",
                    }
                ]
            },
            {
                "decisions": [
                    {
                        "type": "edit",
                        # 使用工具名称和参数编辑操作
                        "edited_action": {
                            # 要调用的工具名称。
                            # 通常与原始操作相同。
                            "name": "new_tool_name",
                            # 要传递给工具的参数。
                            "args": {"key1": "new_value", "key2": "original_value"},
                        },
                    }
                ]
            },
        ],
    )
    # session_id 默认随机的uuid
    session_id: str = str(uuid.uuid4())


class StreamResponse(BaseModel):
    event: Literal[
        "reasoning_token", "token", "tool_calls", "tool_output", "__interrupt__"
    ] = "token"
    data: dict | None = None


def add_general_api_endpoint(
    app: FastAPI,
    agent: CompiledStateGraph,
    path: str = "/api/general_api",
    context: type[BaseModel] | None = None,
):
    """添加通用 API 端点，用于与 LangGraph 交互。

    Parameters
    ----------
    app : FastAPI
        FastAPI 应用实例
    agent : CompiledStateGraph
        编译后的 LangGraph 实例
    path : str, optional
        端点路径， by default "/api/general_api"
    context : type[BaseModel] | None, optional
        上下文对象，表示额外添加的参数， by default None

    Returns
    -------

    Raises
    ------
    ValueError
        query 和 resume 不能同时存在
    """
    # 动态创建 Request 模型
    if context is not None:
        # 如果有 context，创建包含 GeneralAPIRequest 和 context 的组合类
        class Request(GeneralAPIRequest, context):  # type: ignore
            """组合后的请求模型"""

            pass

    else:
        # 如果没有 context，直接使用 GeneralAPIRequest
        Request = GeneralAPIRequest

    @app.post(path, response_model=StreamResponse)
    async def general_api(request: Request):
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

        async def stream_generator():
            text = ""
            async for mode, chunk in agent.astream(
                input=input,
                stream_mode=["messages", "updates"],
                config=config,
                context=context,
            ):
                if mode == "messages":  # 只处理消息流
                    msg, metadata = chunk
                    if metadata.get("tags", []) == ["agent"]:
                        # reasoning_content 部分,reasoning_content 是单个token
                        if msg.additional_kwargs.get("reasoning_content"):
                            # 直接使用reasoning_content作为单个token
                            stream_response.event = "reasoning_token"
                            stream_response.data = {
                                "token": msg.additional_kwargs["reasoning_content"],
                                "id": msg.id,
                            }
                            text += msg.additional_kwargs["reasoning_content"]
                            yield f"data: {stream_response.model_dump_json()}\n\n"
                        if msg.content:
                            # 直接使用msg.content作为单个token
                            stream_response.event = "token"
                            stream_response.data = {"token": msg.content, "id": msg.id}
                            text += msg.content
                            yield f"data: {stream_response.model_dump_json()}\n\n"
                elif mode == "updates":  # 处理更新流
                    # print(f"\n[Update]: {chunk}")

                    if "__interrupt__" in chunk:  # 处理 Human in the Loop
                        stream_response.event = "__interrupt__"
                        stream_response.data = {
                            "__interrupt__": chunk["__interrupt__"][0].value
                        }
                        yield f"data: {stream_response.model_dump_json()}\n\n"

                    if (
                        "model" in chunk
                        and not chunk["model"]["messages"][0].tool_calls
                    ):
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
                        text += (
                            f"\n工具响应： \n{chunk['tools']['messages']}\n{'-'*100}\n"
                        )

            logger.info(f"session_id：{request.session_id} \nFinal Response: \n{text}")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
