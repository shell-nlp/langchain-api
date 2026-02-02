from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
)
from langchain_openai import ChatOpenAI
from langchain_api.middleware import PlanningMiddleware, TestMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langchain_api.settings import settings
from langchain.agents import create_agent


checkpointer = MemorySaver()


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


class Agent:
    def __init__(
        self,
        system_prompt="你是一个善于使用工具的助手。 你每次只能使用一个工具，禁止一次调用多个工具。",
        deep_agent: bool = True,
    ):
        self.model = ChatOpenAI(model=settings.CHAT_MODEL_NAME, tags=["agent"])

        tools = [add, multiply]
        middleware = [TestMiddleware()]

        if deep_agent:
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
            middleware.append(
                PlanningMiddleware(
                    model=ChatOpenAI(
                        model=settings.CHAT_MODEL_NAME,
                    )
                )
            )
            self.agent = create_agent(
                model=self.model,
                tools=tools,
                system_prompt=system_prompt,
                middleware=middleware,
                checkpointer=checkpointer,
            )

    def get_agent(self):
        return self.agent
