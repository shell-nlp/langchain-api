import sys
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.store import BackendContext
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from langchain_api.agent.context import AgentContext
from langchain_api.constant import home_path, workspace_path
from langchain_api.middleware.common import BusinessMiddleware
from langchain_api.settings import settings
from langchain_api.utils import get_chat_model, get_current_time

checkpointer = None
store = None


def init_env():
    global checkpointer, store
    checkpointer = InMemorySaver()
    if settings.PG_DATABASE_URL:
        from langgraph.store.postgres import PostgresStore

        store_ctx = PostgresStore.from_conn_string(settings.PG_DATABASE_URL)
        store = store_ctx.__enter__()
        store.setup()
        logger.info("使用PostgresStore作为长期记忆")
    else:
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()
        logger.info("使用InMemoryStore作为长期记忆")


_platform = sys.platform
if _platform.startswith("win"):
    DEFUALT_SYSTEM_PROMPT = "你的运行环境是 Windows 系统, 你可以使用 Windows 相关的命令"
elif _platform.startswith("linux"):
    DEFUALT_SYSTEM_PROMPT = "你的运行环境是 Linux 系统, 你可以使用 Linux 相关的命令"
elif _platform.startswith("darwin"):
    DEFUALT_SYSTEM_PROMPT = "你的运行环境是 macOS 系统, 你可以使用 macOS 相关的命令"
else:
    DEFUALT_SYSTEM_PROMPT = f"你的运行环境未知: {_platform}"

skills = ["/workspace/skills"]


def user_namespace_factory(ctx: BackendContext[Any, AgentContext]) -> tuple[str, ...]:
    """动态生成用户namespace：('user123', 'filesystem')"""
    user_id = ctx.runtime.context.user_id  # 从context获取
    # TODO 获取config,未来可实现共享命名空间
    # from langchain_core.runnables.config import var_child_runnable_config
    # config = var_child_runnable_config.get()
    return ("filesystem", user_id)  # 用户隔离！


class Agent:
    def __init__(
        self,
        system_prompt=DEFUALT_SYSTEM_PROMPT,
        tools: list = [],
        deep_agent: bool = False,
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.deep_agent = deep_agent
        self.agent = self.init_agent()

    def init_agent(self) -> CompiledStateGraph:
        middleware = []
        if settings.USE_COPILOTKIT:
            from copilotkit import CopilotKitMiddleware

            middleware.append(CopilotKitMiddleware())
        middleware.append(BusinessMiddleware())

        system_prompt = self.system_prompt + get_current_time()
        model = get_chat_model()
        model.tags = ["agent"]
        from langchain_api.tools import get_weather, web_fetch

        backend = None
        tools = self.tools + [get_weather, web_fetch]

        if not workspace_path.exists():
            workspace_path.mkdir(parents=True, exist_ok=True)
        if settings.BACKEND_TYPE == "sandbox":
            from opensandbox.models.sandboxes import Host, Volume

            from langchain_api.backend.open_sandbox import OpenSandbox

            backend = OpenSandbox(
                volumes=[
                    Volume(
                        name="workspace-root",
                        host=Host(path=str(workspace_path)),
                        mount_path="/workspace",
                    )
                ]
            )
            logger.info("使用 OpenSandbox 作为后端")
        elif settings.BACKEND_TYPE == "local_shell":
            from deepagents.backends.local_shell import LocalShellBackend

            backend = LocalShellBackend(
                root_dir=home_path, virtual_mode=True, inherit_env=True
            )
            logger.info("使用 LocalShellBackend 作为后端")
        elif settings.BACKEND_TYPE == "store":
            from langchain_api.agent.utils import copy_skills_to_store

            copy_skills_to_store(skills_dir=workspace_path / "skills", store=store)
            logger.info("使用 StoreBackend 作为后端")

        def make_backend(runtime):
            from deepagents.backends import CompositeBackend, StoreBackend

            nonlocal backend
            if settings.BACKEND_TYPE == "store":
                backend = StoreBackend(runtime, namespace=user_namespace_factory)

            return CompositeBackend(
                default=backend,
                routes={
                    "/memories/": StoreBackend(
                        runtime, namespace=user_namespace_factory
                    ),
                },
            )

        if settings.USE_TOOL_SEARCH:
            from langchain_api.middleware.tool_search import DeferredToolMiddleware

            middleware.append(DeferredToolMiddleware())

        if self.deep_agent:
            logger.info("使用 DeepAgent")

            return create_deep_agent(
                model=model,
                tools=tools,
                system_prompt=system_prompt,
                middleware=middleware,
                backend=make_backend,
                skills=skills,
                checkpointer=checkpointer,
                store=store,
                context_schema=AgentContext,
            )
        else:
            logger.info("正在使用 ReactAgent")

            return create_agent(
                model=model,
                tools=tools,
                system_prompt=system_prompt,
                middleware=middleware,
                checkpointer=checkpointer,
                store=store,
                context_schema=AgentContext,
            )

    def get_agent(self) -> CompiledStateGraph:
        return self.agent


if __name__ == "__main__":
    model = get_chat_model()
    for chunk in model.stream("1+1="):
        print(chunk)
