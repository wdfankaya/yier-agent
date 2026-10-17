"""HTTP API 的请求/响应模型（Pydantic）。"""

from typing import Optional, Annotated

from pydantic import BaseModel, Field, AfterValidator
from app.server.identifiers import validate_id

Identifier = Annotated[str, AfterValidator(validate_id)]


class ConfirmRequest(BaseModel):
    """POST /api/confirm：对 confirm_required 事件做批准 / 拒绝。"""

    session_id: Identifier = Field(..., description="与 /api/chat 同一会话 ID")
    token: str = Field(..., min_length=1, description="confirm_required 事件里的 confirm_token")
    approved: bool = Field(..., description="true=批准并当场执行；false=本会话拒绝该订单")
    user_id: Optional[Identifier] = Field(None, description="可省略；提供时必须与访问令牌身份一致")


class ConfirmResponse(BaseModel):
    """POST /api/confirm 响应。"""

    ok: bool
    approved: Optional[bool] = None
    executed: bool = False
    execution_status: str = "denied"
    order_id: Optional[str] = None
    tool: Optional[str] = None
    amount: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""

    session_id: Identifier = Field(..., description="会话 ID：同一会话的连续轮次传同一值")
    user_id: Optional[Identifier] = Field(None, description="可省略；身份由访问令牌决定")
    message: str = Field(..., min_length=1, max_length=16000, description="用户本轮输入")


class SessionMessagesResponse(BaseModel):
    """GET /api/sessions/{id} 响应：消息列表。"""

    session_id: str
    user_id: str
    history_size: int
    messages: list[dict] = Field(default_factory=list)


class DeleteSessionResponse(BaseModel):
    """DELETE /api/sessions/{id} 响应。"""

    deleted: bool
    session_id: str


class HealthResponse(BaseModel):
    """GET /health 响应。"""

    status: str
    resident_sessions: int
    multi_agent_enabled: Optional[bool] = None
    redis: Optional[str] = None  # ok / down / disabled（Redis 挂了 status 仍为 ok）
