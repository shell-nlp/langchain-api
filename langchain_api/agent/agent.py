from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from loguru import logger

from langchain_api.settings import settings

checkpointer = InMemorySaver()  # 短期记忆
# store = InMemoryStore()  # 长期记忆
store_ctx = PostgresStore.from_conn_string(settings.PG_DATABASE_URL)
store = store_ctx.__enter__()
# store.setup()
shanghai_tz = ZoneInfo("Asia/Shanghai")  # 设置亚洲/上海时区


def get_current_time() -> str:
    # 星期几的映射表
    weekday_map = {
        0: "星期一",
        1: "星期二",
        2: "星期三",
        3: "星期四",
        4: "星期五",
        5: "星期六",
        6: "星期日",
    }
    current_time = datetime.now(shanghai_tz)
    # 获取星期几（0=星期一，6=星期日）
    weekday_num = current_time.weekday()
    weekday_str = weekday_map[weekday_num]
    cur_time = f"""\n当前时间：{current_time.year}年{current_time.month}月{current_time.day}日 星期{weekday_str}"""
    return cur_time


@dataclass
class CustomContext:
    internet_search: bool
    deep_thinking: bool = False


class BusinessMiddleware(AgentMiddleware):
    """业务中间件，用于处理业务相关的逻辑"""

    def wrap_model_call(self, request, handler):
        context: CustomContext = request.runtime.context
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


DEFUALT_SYSTEM_PROMPT = ""
root_dir = Path(__file__).parent.parent.parent

home_path = root_dir / ".langchain_api"
workspace_path = home_path / "workspace"
skills = ["/workspace/skills"]


class Agent:
    def __init__(
        self,
        system_prompt=DEFUALT_SYSTEM_PROMPT,
        tools: list = [],
        deep_agent: bool = False,
    ):
        middleware = []
        system_prompt = system_prompt + get_current_time()
        self.model = ChatDeepSeek(
            model=settings.CHAT_MODEL_NAME,
            api_base=settings.OPENAI_API_BASE,
            api_key=settings.OPENAI_API_KEY,
            tags=["agent"],
            extra_body={"enable_thinking": False},
        )
        from langchain_api.tools.web_fetch import web_fetch

        backend = None
        tools.append(web_fetch)

        if os.getenv("TAVILY_API_KEY"):
            logger.info("TAVILY_API_KEY 已配置，将添加 TavilySearch 工具")
            from langchain_tavily.tavily_search import TavilySearch

            tools.append(TavilySearch())

        middleware = [
            BusinessMiddleware(),
            # HumanInTheLoopMiddleware(
            #     description_prefix="工具执行需要批准",
            #     interrupt_on={
            #         "execute": {"allowed_decisions": ["approve", "reject", "edit"]}
            #     },
            # ),
        ] + middleware

        if not workspace_path.exists():
            workspace_path.mkdir(parents=True, exist_ok=True)
        # 使用沙箱作为后端
        if settings.USE_SANDBOX:
            from opensandbox.models.sandboxes import Host, Volume
            from langchain_api.sandbox.open_sandbox import OpenSandbox

            logger.info("正在使用 OpenSandbox 作为后端")
            backend = OpenSandbox(
                volumes=[
                    Volume(
                        name="workspace-root",
                        host=Host(path=str(workspace_path)),
                        mount_path="/workspace",
                    )
                ]
            )
        else:
            # 使用虚拟文件系统作为后端
            from deepagents.backends.local_shell import LocalShellBackend

            logger.info("正在使用 LocalShellBackend 作为后端")
            backend = LocalShellBackend(root_dir=home_path, virtual_mode=True)

        def make_backend(runtime):
            from deepagents.backends import CompositeBackend, StoreBackend

            return CompositeBackend(
                default=backend,
                routes={"/memories/": StoreBackend(runtime)},  # Persistent storage
            )

        if deep_agent:
            logger.info("正在使用 DeepAgent")

            self.agent = create_deep_agent(
                model=self.model,
                tools=tools,
                system_prompt=system_prompt,
                middleware=middleware,
                backend=make_backend,
                skills=skills,
                checkpointer=checkpointer,
                store=store,
            )
        else:
            logger.info("正在使用 ReactAgent")

            self.agent = create_agent(
                model=self.model,
                tools=tools,
                system_prompt=system_prompt,
                middleware=middleware,
                checkpointer=checkpointer,
                skills=skills,
                store=store,
            )

    def get_agent(self):
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
