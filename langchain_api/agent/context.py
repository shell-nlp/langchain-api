import uuid

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    internet_search: bool = Field(False, description="是否允许使用互联网搜索")
    deep_thinking: bool = Field(False, description="是否启用深度思考")
    # 会话ID，用于跟踪用户会话
    session_id: str = Field(str(uuid.uuid4()), description="会话ID")
    # 用户ID，用于标识用户
    user_id: str = Field("default", description="用户ID")
