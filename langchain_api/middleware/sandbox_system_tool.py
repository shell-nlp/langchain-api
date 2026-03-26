from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables.config import var_child_runnable_config

from langchain_api.tools.sandbox import (
    edit_file,
    execute_tool,
    glob_tool,
    grep_tool,
    ls_tool,
    read_file,
    write_file,
)
from langchain_core.runnables.config import RunnableConfig


class SandboxSystemToolMiddleware(AgentMiddleware):
    """沙箱系统工具中间件"""

    tools = [
        execute_tool,
        ls_tool,
        read_file,
        write_file,
        grep_tool,
        glob_tool,
        edit_file,
    ]

    def after_agent(self, state, runtime):
        config: RunnableConfig = var_child_runnable_config.get()
        return None
