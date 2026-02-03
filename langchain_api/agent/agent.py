import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_api.middleware import PlanningMiddleware, CallBackMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langchain_api.settings import settings
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from pydantic import Field
from loguru import logger

checkpointer = MemorySaver()


def eval_tool(expression: str = Field(..., description="要计算的数学表达式")) -> float:
    """用来计算数学表达式的工具。输入一个数学表达式，返回计算结果。"""
    try:
        return float(eval(expression))
    except Exception:
        raise ValueError("无法计算表达式")


class Agent:
    def __init__(
        self,
        system_prompt="你是一个善于使用工具的助手。 你每次只能使用一个工具，禁止一次调用多个工具。",
        tools: list = [],
        middleware: List[AgentMiddleware] = [],
        deep_agent: bool = False,
    ):
        self.model = ChatOpenAI(model=settings.CHAT_MODEL_NAME, tags=["agent"])
        tools = [eval_tool] + tools
        if os.getenv("TAVILY_API_KEY"):
            logger.info("TAVILY_API_KEY 已配置，将添加 TavilySearch 工具")
            from langchain_tavily.tavily_search import TavilySearch

            tools.append(TavilySearch())

        middleware = [CallBackMiddleware()] + middleware

        if deep_agent:
            logger.info("正在使用 DeepAgent")
            from deepagents import create_deep_agent
            from deepagents.backends import FilesystemBackend

            self.agent = create_deep_agent(
                model=self.model,
                tools=tools,
                system_prompt=system_prompt,
                middleware=middleware,
                backend=FilesystemBackend(root_dir=".", virtual_mode=True),
            )
        else:
            logger.info("正在使用 ReactAgent")
            # middleware.append(
            #     PlanningMiddleware(
            #         model=ChatOpenAI(
            #             model=settings.CHAT_MODEL_NAME,
            #         )
            #     ),
            # )
            self.agent = create_agent(
                model=self.model,
                tools=tools,
                system_prompt=system_prompt,
                middleware=middleware,
                checkpointer=checkpointer,
            )

    def get_agent(self):
        return self.agent
