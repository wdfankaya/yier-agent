"""敏感操作 HITL 确认闸。

敏感工具在执行前检查用户授权，确认状态由会话独立管理。
本闸挂在 ToolManager 上，按会话隔离：
- 未授权：不执行，返回 confirm_required + token（SSE 另推事件）
- 授权后：仅允许已展示参数执行一次；同参数重试返回记录的结果
- 拒绝后：本会话该订单进黑名单，直接 denied

HTTP `POST /api/confirm` 或用户短回复「确认」/「拒绝」都会改闸门。
"""

from __future__ import annotations

import json
import re
import uuid
import time
import threading
from copy import deepcopy
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
    """Immutable, expiring, single-use approvals for exact tool arguments."""

    def __init__(self):
        self.pending = {}
        self.denied_orders = set()
        self._grants = set()
        self._results = {}
        self._lock = threading.RLock()

    @staticmethod
    def key(name, arguments):
        return json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)

    def reset(self):
        with self._lock:
            self.pending.clear()
            self.denied_orders.clear()
            self._grants.clear()
            self._results.clear()

    def _expire(self):
        for token, payload in list(self.pending.items()):
            if payload["expires"] <= time.monotonic():
                del self.pending[token]

    def intercept(self, name, arguments):
        from app.config.settings import settings
        if not settings.hitl_enabled:
            return None
        with self._lock:
            self._expire()
            key = self.key(name, arguments)
            order_id = str(arguments.get("order_id") or "")
            if key in self._results:
                return deepcopy(self._results[key])
            if key in self._grants:
                self._grants.remove(key)
                # Until recorded, retries must not issue another side effect.
                self._results[key] = {"status": "execution_unknown", "message": "操作正在执行或结果未知，请核实状态，不要重复提交"}
                return None
            if order_id in self.denied_orders:
                return {"status": "denied", "tool": name, "order_id": order_id}
            for token, payload in list(self.pending.items()):
                if payload["tool"] == name and payload["order_id"] == order_id:
                    if payload["key"] == key:
                        return self._event(token, payload)
                    # Changed arguments invalidate the previously displayed button.
                    del self.pending[token]
            token = uuid.uuid4().hex
            payload = {"tool": name, "args": deepcopy(arguments), "order_id": order_id,
                       "amount": ORDERS.get(order_id, {}).get("total"), "key": key,
                       "expires": time.monotonic() + settings.hitl_token_ttl}
            self.pending[token] = payload
            return self._event(token, payload)

    @staticmethod
    def _event(token, payload):
        return {"status": "confirm_required", "confirm_token": token,
                **{k: deepcopy(payload[k]) for k in ("tool", "args", "order_id", "amount")}}

    def resolve(self, token, approved):
        with self._lock:
            self._expire()
            payload = self.pending.pop(token, None)
            if payload is None:
                return {"ok": False, "error": "确认令牌无效、过期或已使用；服务重启后需重新发起确认"}
            if approved:
                self._grants.add(payload["key"])
            else:
                self.denied_orders.add(payload["order_id"])
            return {"ok": True, "approved": approved,
                    **{k: deepcopy(payload[k]) for k in ("tool", "args", "order_id", "amount")}}

    def record(self, name, arguments, result):
        with self._lock:
            key = self.key(name, arguments)
            if key in self._results:
                self._results[key] = deepcopy(result)

    def utterance(self, text):
        with self._lock:
            self._expire()
            # A bare yes must never select an arbitrary order.
            if len(self.pending) != 1:
                return None
            t = (text or "").strip()
            if _YES.fullmatch(t):
                return next(iter(self.pending)), True
            if _NO.fullmatch(t):
                return next(iter(self.pending)), False
            return None

    def consume_utterance(self, text):
        choice = self.utterance(text)
        if choice is None:
            return None
        info = self.resolve(*choice)
        return ("approved" if choice[1] else "denied") if info["ok"] else None


def resolve_utterance_on_agent(agent, text):
    choice = agent.hitl.utterance(text)
    if choice is not None:
        return resolve_on_agent(agent, *choice)


async def aresolve_utterance_on_agent(agent, text):
    choice = agent.hitl.utterance(text)
    if choice is not None:
        return await aresolve_on_agent(agent, *choice)


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
    from app.agent.async_utils import run_sync
    return run_sync(aresolve_on_agent(agent, token, approved))


async def aresolve_on_agent(agent, token: str, approved: bool) -> dict:
    """All confirmations use the same bounded async tool executor as chat."""
    import asyncio
    info = agent.hitl.resolve(token, approved)
    if not info.get("ok"):
        return info
    result = None
    status = "denied"
    if approved:
        tm = _tool_manager_for(agent, info["tool"])
        raw = await tm.aexecute_tool(info["tool"], info["args"])
        try:
            result = json.loads(raw)
        except (ValueError, TypeError):
            result = {"status": "execution_unknown"}
        if not isinstance(result, dict):
            result = {"status": "execution_unknown"}
        if result.get("status") == "execution_unknown":
            status = "unknown"
        elif result.get("error") or result.get("success") is False or result.get("status") in ("denied", "confirm_required"):
            status = "failed"
        else:
            status = "succeeded"
        agent.raw_messages.extend([
            {"role": "user", "content": f"[HITL] 已确认 {info['tool']} {info['order_id']}"},
            {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
        ])
    else:
        agent.raw_messages.append({"role": "user", "content": f"[HITL] 已拒绝 {info['tool']} {info['order_id']}"})
    if hasattr(agent, "save"):
        await asyncio.to_thread(agent.save)
    return {**info, "executed": status == "succeeded", "execution_status": status, "result": result}
