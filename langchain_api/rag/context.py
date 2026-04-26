from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Stable context for a single RAG request."""

    user_id: str = Field("default", description="User ID")
    index_name: str = Field("", description="Passage index name")
    graph_name: str = Field("", description="Graph index prefix")
    internet_search: bool = Field(False, description="Enable internet search")
    deep_thinking: bool = Field(False, description="Enable deep thinking")
