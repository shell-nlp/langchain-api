import os
import sys
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.store import BackendContext
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from langchain_api.agent.context import AgentContext
from langchain_api.agent.state import StateSchema
from langchain_api.constant import home_path, workspace_path
from langchain_api.settings import settings
from langchain_api.utils import get_chat_model, get_current_time

checkpointer = InMemorySaver()  # 短期记忆
if settings.PG_DATABASE_URL:
    from langgraph.store.postgres import PostgresStore

    store_ctx = PostgresStore.from_conn_string(settings.PG_DATABASE_URL)
    store = store_ctx.__enter__()
    store.setup()
    logger.info("使用PostgresStore作为长期记忆")
else:
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()  # 长期记忆
    logger.info("使用InMemoryStore作为长期记忆")
# 检查当前操作系统是Windows还是Linux
platform = sys.platform
if platform.startswith("win"):
    DEFUALT_SYSTEM_PROMPT = "你的运行环境是 Windows 系统, 你可以使用 Windows 相关的命令"
elif platform.startswith("linux"):
    DEFUALT_SYSTEM_PROMPT = "你的运行环境是 Linux 系统, 你可以使用 Linux 相关的命令"
elif platform.startswith("darwin"):
    DEFUALT_SYSTEM_PROMPT = "你的运行环境是 macOS 系统, 你可以使用 macOS 相关的命令"
else:
    DEFUALT_SYSTEM_PROMPT = f"你的运行环境未知: {platform}"

skills = ["/workspace/skills"]


# https://github.com/CopilotKit/CopilotKit/issues/2646
class BusinessMiddleware(AgentMiddleware[None, AgentContext, None]):
    """业务中间件，用于处理业务相关的逻辑"""

    state_schema = StateSchema

    def wrap_model_call(self, request, handler):
        context: AgentContext = request.runtime.context
        internet_search = context.internet_search
        deep_thinking = context.deep_thinking
        if not internet_search:
            # 禁用互联网搜索相关的工具调用
            filtered_tools = [
                tool for tool in request.tools if tool.name != "tavily_search"
            ]
            request = request.override(tools=filtered_tools)

        # 处理深度思考
        if deep_thinking:
            # 为模型调用添加深度思考参数
            if hasattr(request, "model_settings"):
                model_settings = request.model_settings.copy()
            else:
                model_settings = {}
            model_settings["extra_body"] = model_settings.get("extra_body", {})
            model_settings["extra_body"]["enable_thinking"] = True
            request = request.override(model_settings=model_settings)

        return handler(request)

    async def awrap_model_call(self, request, handler):
        return await self.wrap_model_call(request, handler)


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
        middleware = []
        # 是否使用CopilotKit中间件
        if settings.USE_COPILOTKIT:
            from copilotkit import CopilotKitMiddleware

            middleware.append(CopilotKitMiddleware())
        # else:
        middleware.append(BusinessMiddleware())
        middleware.extend(
            [
                # HumanInTheLoopMiddleware(
                #     description_prefix="工具执行需要批准",
                #     interrupt_on={
                #         "execute": {"allowed_decisions": ["approve", "reject", "edit"]}
                #     },
                # ),
            ]
        )

        system_prompt = system_prompt + get_current_time()
        self.model = get_chat_model()
        self.model.tags = ["agent"]
        from langchain_api.tools import get_weather, web_fetch

        backend = None
        tools.extend([get_weather, web_fetch])

        if os.getenv("TAVILY_API_KEY"):
            logger.info("TAVILY_API_KEY 已配置，将添加 TavilySearch 工具")
            from langchain_tavily.tavily_search import TavilySearch

            tools.append(TavilySearch())

        if not workspace_path.exists():
            workspace_path.mkdir(parents=True, exist_ok=True)
        # 使用沙箱作为后端
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
            # 使用虚拟文件系统作为后端
            from deepagents.backends.local_shell import LocalShellBackend

            backend = LocalShellBackend(
                root_dir=home_path, virtual_mode=True, inherit_env=True
            )
            logger.info("使用 LocalShellBackend 作为后端")

        # -------------------------------
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
                },  # Persistent storage
            )

        if settings.USE_TOOL_SEARCH:
            # 最后加载,保证工具可以被完全拦截
            from langchain_api.middleware.tool_search import DeferredToolMiddleware

            middleware.append(DeferredToolMiddleware())

        if deep_agent:
            logger.info("使用 DeepAgent")

            self.agent = create_deep_agent(
                model=self.model,
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

            self.agent = create_agent(
                model=self.model,
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
    model.get_num_tokens_from_messages
    for chunk in model.stream("1+1="):
        print(chunk)
