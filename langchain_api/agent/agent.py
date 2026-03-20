import os
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.store import BackendContext
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from langchain_api.agent.context import AgentContext
from langchain_api.settings import settings
from langchain_api.utils import get_current_time


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

DEFUALT_SYSTEM_PROMPT = ""
root_dir = Path(__file__).parent.parent.parent

home_path = root_dir / ".langchain_api"
workspace_path = home_path / "workspace"
skills = ["/workspace/skills"]
# TODO 开启后会导致smith 异常


class BusinessMiddleware(AgentMiddleware):
    """业务中间件，用于处理业务相关的逻辑"""

    def wrap_model_call(self, request, handler):
        context: AgentContext = request.runtime.context
        logger.info(context)
        # tool_names = [tool.name for tool in request.tools]
        # logger.info(tool_names)
        if not context.internet_search:
            # 禁用互联网搜索相关的工具调用
            filtered_tools = [
                tool for tool in request.tools if tool.name != "tavily_search"
            ]
            request = request.override(tools=filtered_tools)

        # 处理深度思考
        if context.deep_thinking:
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
    return (user_id, "filesystem")  # 用户隔离！


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
        else:
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
        self.model = ChatDeepSeek(
            model=settings.CHAT_MODEL_NAME,
            api_base=settings.OPENAI_API_BASE,
            api_key=settings.OPENAI_API_KEY,
            tags=["agent"],
            extra_body={"enable_thinking": False},
        )
        from langchain_api.tools import web_fetch, get_weather

        backend = None
        tools.extend([get_weather, web_fetch])

        if os.getenv("TAVILY_API_KEY"):
            logger.info("TAVILY_API_KEY 已配置，将添加 TavilySearch 工具")
            from langchain_tavily.tavily_search import TavilySearch

            tools.append(TavilySearch())

        if not workspace_path.exists():
            workspace_path.mkdir(parents=True, exist_ok=True)
        # 使用沙箱作为后端
        if settings.USE_SANDBOX:
            from opensandbox.models.sandboxes import Host, Volume
            from langchain_api.sandbox.open_sandbox import OpenSandbox

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
        else:
            # 使用虚拟文件系统作为后端
            from deepagents.backends.local_shell import LocalShellBackend

            backend = LocalShellBackend(root_dir=home_path, virtual_mode=True)
            logger.info("使用 LocalShellBackend 作为后端")

        def make_backend(runtime):
            from deepagents.backends import CompositeBackend, StoreBackend

            return CompositeBackend(
                default=backend,
                routes={
                    "/memories/": StoreBackend(
                        runtime, namespace=user_namespace_factory
                    )
                },  # Persistent storage
            )

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
    model = ChatDeepSeek(
        model=settings.CHAT_MODEL_NAME,
        tags=["agent"],
        api_base=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
        extra_body={"enable_thinking": True},
    )
    model.get_num_tokens_from_messages
    for chunk in model.stream("1+1="):
        print(chunk)
