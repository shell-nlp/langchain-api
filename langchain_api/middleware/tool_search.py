import re
from typing import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ExtendedModelResponse
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.types import Command
from loguru import logger

MAX_RESULTS = 3


def _regex_score(pattern: str, tool: dict) -> int:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    return len(regex.findall(f"{tool['name']} {tool['description']}"))


def deferred_tool_search(deferred_tool: list[dict], query: str) -> list[dict]:
    """按正则表达式模式搜索延迟工具，匹配名称和描述。支持三种查询形式（与 Claude Code 保持一致）：
    "select:name1,name2" — 精确匹配名称
    "+keyword rest" — 名称必须包含关键词，按 rest 排序
    "keyword query" — 对名称和描述进行正则表达式匹配
    返回：
    匹配的 BaseTool 对象列表"""
    if query.startswith("select:"):
        names = {n.strip() for n in query[7:].split(",")}
        return [tool for tool in deferred_tool if tool["name"] in names][:MAX_RESULTS]
    if query.startswith("+"):
        parts = query[1:].split(None, 1)
        required = parts[0].lower()
        candidates = [
            tool for tool in deferred_tool if required in tool["name"].lower()
        ]
        if len(parts) > 1:
            candidates.sort(
                key=lambda tool: _regex_score(parts[1], tool),
                reverse=True,
            )
        return [tool for tool in candidates][:MAX_RESULTS]

    # General regex search
    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(query), re.IGNORECASE)

    scored = []
    for tool in deferred_tool:
        searchable = f"{tool['name']} {tool['description']}"
        if regex.search(searchable):
            score = 2 if regex.search(tool["name"]) else 1
            scored.append((score, tool))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [tool for _, tool in scored][:MAX_RESULTS]


class ToolSearchState(AgentState):
    deferred_tool: NotRequired[list[dict]]


@tool
def tool_search(query: str, runtime: ToolRuntime) -> dict:
    """获取延迟工具完整架构定义，使其可被调用。
    延迟工具会在系统提示词的 <available-deferred-tools> 中显示名称。在获取之前，仅知道其名称——没有参数架构，因此无法调用该工具。此工具接收查询，将其与延迟工具列表进行匹配，并返回匹配工具的完整定义。一旦工具的架构出现在该结果中，它即可被调用。
    查询形式：
    "select:Read,Edit,Grep" —— fetch these exact tools by name
    "notebook jupyter" —— keyword search, up to max_results best matches
    "+slack send" —— require "slack" in the name, rank by remaining terms
    参数：
    query: 用于查找延迟工具的查询。使用 "select:<tool_name>" 进行直接选择，或使用关键词进行搜索。
    返回：
    匹配的工具定义，以 JSON 数组形式返回。"""
    deferred_tool = runtime.state.get("deferred_tool", None)
    matched_tools = deferred_tool_search(deferred_tool, query)
    if not matched_tools:
        return f"未找到匹配的工具: {query}"
    tool_defs = matched_tools[:MAX_RESULTS]

    return Command(
        update={
            "messages": [
                ToolMessage(content=str(tool_defs), tool_call_id=runtime.tool_call_id)
            ],
        }
    )


class DeferredToolMiddleware(AgentMiddleware):
    """延迟工具中间件"""

    tools = [tool_search]
    state_schema = ToolSearchState

    def wrap_model_call(self, request, handler):
        state = request.state
        current_tools = request.tools  # 原始是有全部的工具
        current_tool_names = [
            tool.name for tool in current_tools if tool.name != tool_search.name
        ]
        available_deferred_tools = "\n".join(current_tool_names)
        deferred_tool_prompt = f"\n<available-deferred-tools>\n{available_deferred_tools}\n</available-deferred-tools>\n"
        new_system_prpmpt = request.system_prompt + deferred_tool_prompt
        deferred_tool = state.get("deferred_tool", None)
        update = None
        if deferred_tool is None:
            deferred_tools = [
                tool for tool in current_tools if tool.name != tool_search.name
            ]

            logger.info(f"添加 {len(deferred_tools)} 个延迟工具到状态")
            request = request.override(
                system_prompt=new_system_prpmpt, tools=[tool_search]
            )
            update = {
                "deferred_tool": [
                    convert_to_openai_function(tool) for tool in deferred_tools
                ]
            }
        else:
            request = request.override(
                system_prompt=new_system_prpmpt, tools=[tool_search]
            )
        return ExtendedModelResponse(
            model_response=handler(request),
            command=Command(update=update),
        )

    async def awrap_model_call(self, request, handler):
        state = request.state
        current_tools = request.tools  # 原始是有全部的工具
        current_tool_names = [
            tool.name for tool in current_tools if tool.name != tool_search.name
        ]
        available_deferred_tools = "\n".join(current_tool_names)
        deferred_tool_prompt = f"\n<available-deferred-tools>\n{available_deferred_tools}\n</available-deferred-tools>\n"
        new_system_prpmpt = request.system_prompt + deferred_tool_prompt
        deferred_tool = state.get("deferred_tool", None)
        update = None
        if deferred_tool is None:
            deferred_tools = [
                tool for tool in current_tools if tool.name != tool_search.name
            ]

            logger.info(f"添加 {len(deferred_tools)} 个延迟工具到状态")
            request = request.override(
                system_prompt=new_system_prpmpt, tools=[tool_search]
            )
            update = {
                "deferred_tool": [
                    convert_to_openai_function(tool) for tool in deferred_tools
                ]
            }
        else:
            request = request.override(
                system_prompt=new_system_prpmpt, tools=[tool_search]
            )
        return ExtendedModelResponse(
            model_response=await handler(request),
            command=Command(update=update),
        )
