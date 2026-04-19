import os

from langchain.agents.middleware import AgentMiddleware
from loguru import logger

from langchain_api.agent.context import AgentContext
from langchain_api.agent.state import StateSchema


# https://github.com/CopilotKit/CopilotKit/issues/2646
class BusinessMiddleware(AgentMiddleware[None, AgentContext, None]):
    """业务中间件，用于处理业务相关的逻辑"""

    state_schema = StateSchema

    def __init__(self) -> None:
        self.tools = []
        if os.getenv("TAVILY_API_KEY"):
            logger.info("TAVILY_API_KEY 已配置，将添加 TavilySearch 工具")
            from langchain_tavily.tavily_search import TavilySearch

            self.tools.append(TavilySearch())

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
        if hasattr(request, "model_settings"):
            model_settings = request.model_settings.copy()
        else:
            model_settings = {}
        if deep_thinking:
            # 为模型调用添加深度思考参数
            model_settings["extra_body"] = model_settings.get("extra_body", {})
            model_settings["extra_body"]["enable_thinking"] = True
            request = request.override(model_settings=model_settings)
        else:
            # 移除深度思考参数
            model_settings["extra_body"] = model_settings.get("extra_body", {})
            model_settings["extra_body"]["enable_thinking"] = False
            request = request.override(model_settings=model_settings)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        return await self.wrap_model_call(request, handler)
