from langchain.agents.middleware import AgentState

from typing import NotRequired


class StateSchema(AgentState):
    internet_search: NotRequired[bool]
    deep_thinking: NotRequired[bool]
