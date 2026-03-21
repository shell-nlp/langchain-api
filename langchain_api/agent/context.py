import uuid

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    # 会话ID，用于跟踪用户会话
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="会话ID"
    )
    # 用户ID，用于标识用户
    user_id: str = Field("default", description="用户ID")
