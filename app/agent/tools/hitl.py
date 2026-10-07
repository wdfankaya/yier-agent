"""敏感操作 HITL 确认闸。

敏感工具在执行前检查用户授权，确认状态由会话独立管理。
本闸挂在 ToolManager 上，按会话隔离：
- 未授权：不执行，返回 confirm_required + token（SSE 另推事件）
- 授权后：本会话该订单进白名单，不再二次确认
- 拒绝后：本会话该订单进黑名单，直接 denied

HTTP `POST /api/confirm` 或用户短回复「确认」/「拒绝」都会改闸门。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from app.agent.tools.mock_data import ORDERS

_YES = re.compile(
    r"^(好的?|确认|同意|可以|是的|批准|ok|yes)[。.!！]*$",
    re.IGNORECASE,
)
_NO = re.compile(
    r"^(拒绝|取消|不要|不行|不同意|no)[。.!！]*$",
    re.IGNORECASE,
)


class ConfirmationGate:
    """一个会话一把闸。"""

    def __init__(self) -> None:
        self.pending: dict[str, dict] = {}
        self.allowed_orders: set[str] = set()
        self.denied_orders: set[str] = set()

    def reset(self) -> None:
        self.pending.clear()
        self.allowed_orders.clear()
        self.denied_orders.clear()

    def intercept(self, name: str, arguments: dict) -> Optional[dict]:
        """需要拦住时返回给 LLM 的 dict；放行返回 None。"""
        from app.config.settings import settings
        if not settings.hitl_enabled:
            return None
        order_id = str(arguments.get("order_id") or "")
        if order_id in self.denied_orders:
            return {
                "status": "denied",
                "tool": name,
                "order_id": order_id,
                "message": "您已拒绝本会话内对该订单的退款。如需重新办理，请新开一个会话。",
            }
        if order_id in self.allowed_orders:
            return None
        for tok, p in self.pending.items():
            if p.get("order_id") == order_id and p.get("tool") == name:
                p["args"] = dict(arguments)
                return {
                    "status": "confirm_required",
                    "tool": name,
                    "args": dict(arguments),
                    "confirm_token": tok,
                    "order_id": order_id,
                    "amount": p.get("amount"),
                }
        token = uuid.uuid4().hex
        amount = None
        order = ORDERS.get(order_id)
        if order:
            amount = order.get("total")
        payload = {
            "tool": name,
            "args": dict(arguments),
            "order_id": order_id,
            "amount": amount,
        }
        self.pending[token] = payload
        return {
            "status": "confirm_required",
            "tool": name,
            "args": dict(arguments),
            "confirm_token": token,
            "order_id": order_id,
            "amount": amount,
        }

    def resolve(self, token: str, approved: bool) -> dict:
        pending = self.pending.pop(token, None)
        if pending is None:
            return {"ok": False, "error": "确认令牌无效或已使用"}
        order_id = pending["order_id"]
        if approved:
            self.allowed_orders.add(order_id)
            self.denied_orders.discard(order_id)
        else:
            self.denied_orders.add(order_id)
            self.allowed_orders.discard(order_id)
        return {
            "ok": True,
            "approved": approved,
            "order_id": order_id,
            "tool": pending["tool"],
            "args": pending["args"],
            "amount": pending.get("amount"),
        }

    def consume_utterance(self, text: str) -> Optional[str]:
        """有待确认时，短回复「确认」/「拒绝」直接改闸。返回 approved/denied，否则 None。"""
        if not self.pending:
            return None
        t = (text or "").strip()
        tok = next(iter(self.pending))
        if _YES.match(t):
            self.resolve(tok, True)
            return "approved"
        if _NO.match(t):
            self.resolve(tok, False)
            return "denied"
        return None


def notify_confirm_required(
    on_event, emit, result_str: str, cli_prefix: str = "",
) -> None:
    """工具结果若是待确认：SSE 多推一帧 confirm_required；CLI 打印操作提示。"""
    payload = parse_hitl_payload(result_str)
    if not payload:
        return
    if payload.get("status") == "confirm_required":
        if on_event is not None:
            emit(
                "confirm_required",
                tool=payload.get("tool"),
                args=payload.get("args"),
                confirm_token=payload.get("confirm_token"),
                order_id=payload.get("order_id"),
                amount=payload.get("amount"),
            )
            return
        amt = payload.get("amount")
        amt_s = f"¥{amt}" if amt is not None else "金额未知"
        print(
            f"{cli_prefix}⏳ [待确认] {payload.get('tool')} 订单 "
            f"{payload.get('order_id')} {amt_s} — 回复「确认」或「拒绝」"
        )
        return
    if payload.get("status") == "denied" and on_event is None:
        print(
            f"{cli_prefix}🚫 [已拒绝] 本会话不再执行 {payload.get('tool')} "
            f"订单 {payload.get('order_id')}"
        )


def parse_hitl_payload(result_str: str) -> Optional[dict]:
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and data.get("status") in ("confirm_required", "denied"):
        return data
    return None


def _tool_manager_for(agent, tool_name: str):
    tm = getattr(agent, "tool_manager", None)
    if tm is not None:
        return tm
    for sub in getattr(agent, "agents", {}).values():
        src = getattr(sub.tool_manager, "_tool_source", {})
        if tool_name in src:
            return sub.tool_manager
    agents = getattr(agent, "agents", None)
    if agents:
        return next(iter(agents.values())).tool_manager
    raise RuntimeError("找不到可执行该工具的 ToolManager")


def resolve_on_agent(agent, token: str, approved: bool) -> dict:
    """HTTP / 测试入口：改闸；批准则当场执行敏感工具并写入对话历史。"""
    gate = agent.hitl
    info = gate.resolve(token, approved)
    if not info.get("ok"):
        return info
    result = None
    if approved:
        tm = _tool_manager_for(agent, info["tool"])
        raw = tm.execute_tool(info["tool"], info["args"])
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"raw": raw}
        agent.raw_messages.append({
            "role": "user",
            "content": f"[HITL] 已确认 {info['tool']} {info['order_id']}",
        })
        agent.raw_messages.append({
            "role": "assistant",
            "content": json.dumps(result, ensure_ascii=False),
        })
    else:
        agent.raw_messages.append({
            "role": "user",
            "content": f"[HITL] 已拒绝 {info['tool']} {info['order_id']}",
        })
    if hasattr(agent, "save"):
        agent.save()
    return {**info, "executed": bool(approved), "result": result}
