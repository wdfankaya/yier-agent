"""HITL 确认流。

1. 敏感工具拦截 → confirm_required，未授权不执行
2. 同一订单复用 token；授权后本会话放行
3. 另一订单仍要确认
4. 拒绝后本会话 denied
5. hitl_enabled=False 直接执行
6. 短回复「确认」改闸；「我确认要退货」在有 pending 时也不误触
7. POST /api/confirm 批准当场执行；坏 token 400；无会话 404
8. GET /demo 调试页

用法：python tests/test_hitl.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.agent.tools.hitl import ConfirmationGate, resolve_on_agent  # noqa: E402
from app.agent.tools.manager import ToolManager  # noqa: E402
from app.config.settings import settings  # noqa: E402

ORDER_A = "ORD-20240120-002"  # pending，可退
ORDER_B = "ORD-20240115-001"
REASON = {"order_id": ORDER_A, "reason": "不想要了"}


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _tm(gate: ConfirmationGate) -> ToolManager:
    return ToolManager(use_mcp=False, hitl=gate)


def _refund(tm: ToolManager, order_id: str = ORDER_A) -> dict:
    raw = tm.execute_tool("apply_refund", {"order_id": order_id, "reason": "不想要了"})
    return json.loads(raw)


class _HitlOn:
    def __enter__(self):
        self._old = settings.hitl_enabled
        settings.hitl_enabled = True
        return self

    def __exit__(self, *args):
        settings.hitl_enabled = self._old


def test_intercept_blocks():
    print("\n[1/8] 未授权拦截，不真正退款")
    with _HitlOn():
        gate = ConfirmationGate()
        tm = _tm(gate)
        data = _refund(tm)
        if data.get("status") != "confirm_required":
            _fail(f"期望 confirm_required，实际 {data}")
        if "confirm_token" not in data or data.get("order_id") != ORDER_A:
            _fail(f"缺 token/订单号: {data}")
        if data.get("amount") != 1828.90:
            _fail(f"金额应对齐 mock total，实际 {data.get('amount')}")
        again = _refund(tm)
        if again.get("confirm_token") != data["confirm_token"]:
            _fail("同一订单应复用 token")
        if "success" in data:
            _fail("拦截结果不应带 success（尚未执行）")
        _ok("拦截返回 confirm_required，复用 token，未执行退款")


def test_allow_then_other_order():
    print("\n[2/8] 授权后本单放行，另一单仍要确认")
    with _HitlOn():
        gate = ConfirmationGate()
        tm = _tm(gate)
        tok = _refund(tm)["confirm_token"]
        info = gate.resolve(tok, True)
        if not info["ok"] or ORDER_A not in gate.allowed_orders:
            _fail(f"resolve 失败 {info}")
        done = _refund(tm)
        if not done.get("success"):
            _fail(f"白名单后应真正退款，实际 {done}")
        other = _refund(tm, ORDER_B)
        if other.get("status") != "confirm_required":
            _fail(f"另一订单仍应拦截，实际 {other}")
        _ok("本单免二次确认；其他订单仍拦截")


def test_deny():
    print("\n[3/8] 拒绝后本会话 denied")
    with _HitlOn():
        gate = ConfirmationGate()
        tm = _tm(gate)
        tok = _refund(tm)["confirm_token"]
        gate.resolve(tok, False)
        data = _refund(tm)
        if data.get("status") != "denied":
            _fail(f"期望 denied，实际 {data}")
        _ok("拒绝后再次调用直接 denied")


def test_disabled_skips():
    print("\n[4/8] hitl_enabled=False 直接执行")
    old = settings.hitl_enabled
    settings.hitl_enabled = False
    try:
        gate = ConfirmationGate()
        tm = _tm(gate)
        data = _refund(tm)
        if not data.get("success"):
            _fail(f"关闭 HITL 应直接退款，实际 {data}")
        if gate.pending:
            _fail("关闭 HITL 不应写入 pending")
        _ok("开关关闭则 apply_refund 立即执行")
    finally:
        settings.hitl_enabled = old


def test_utterance():
    print("\n[5/8] 短回复「确认」改闸；长句不误触")
    with _HitlOn():
        gate = ConfirmationGate()
        if gate.consume_utterance("确认") is not None:
            _fail("无 pending 时不应消费「确认」")
        if gate.consume_utterance("我确认要退货") is not None:
            _fail("无 pending 时长句不应当 HITL")
        tm = _tm(gate)
        _refund(tm)
        if gate.consume_utterance("我确认要退货") is not None:
            _fail("有 pending 时「我确认要退货」也不应整句匹配")
        if gate.consume_utterance("确认") != "approved":
            _fail("短回复「确认」应放行")
        if ORDER_A not in gate.allowed_orders:
            _fail("确认后应进白名单")
        if _refund(tm).get("success") is not True:
            _fail("确认后本轮应能真正退款")
        gate2 = ConfirmationGate()
        _tm(gate2).execute_tool("apply_refund", REASON)
        if gate2.consume_utterance("拒绝") != "denied":
            _fail("短回复「拒绝」应拉黑")
        _ok("短回复改闸；「我确认要退货」不误当作授权")


def test_resolve_on_agent():
    print("\n[6/8] resolve_on_agent 批准当场执行并写入历史")
    with _HitlOn():
        from app.agent.chat import YierAgent

        old = (
            settings.db_enabled,
            settings.redis_enabled,
            settings.memory_enabled,
            settings.mcp_enabled,
        )
        settings.db_enabled = False
        settings.redis_enabled = False
        settings.memory_enabled = False
        settings.mcp_enabled = False
        sid = uuid.uuid4().hex[:8]
        path = str(ROOT / "app" / "sessions" / "server" / "hitl-test" / f"{sid}.json")
        agent = None
        try:
            agent = YierAgent(session_path=path, user_id="hitl-u")
            tok = json.loads(
                agent.tool_manager.execute_tool("apply_refund", REASON)
            )["confirm_token"]
            info = resolve_on_agent(agent, tok, True)
            if not info.get("executed") or not (info.get("result") or {}).get("success"):
                _fail(f"批准应当场退款，实际 {info}")
            joined = " ".join(m.get("content", "") for m in agent.raw_messages)
            if "[HITL] 已确认" not in joined:
                _fail("历史应留下 HITL 确认痕迹")
            bad = resolve_on_agent(agent, "no-such-token", True)
            if bad.get("ok"):
                _fail("坏 token 应 ok=False")
            _ok("批准即执行；坏 token 被拒")
        finally:
            (
                settings.db_enabled,
                settings.redis_enabled,
                settings.memory_enabled,
                settings.mcp_enabled,
            ) = old
            try:
                agent.close()
            except Exception:
                pass


def test_http_confirm():
    print("\n[7/8] POST /api/confirm 批准执行；坏 token 400；无会话 404")
    from fastapi.testclient import TestClient
    from app.agent.chat import YierAgent
    from app.server.main import app, registry
    from app.server.ratelimit import reset_limiter

    old = (
        settings.db_enabled,
        settings.redis_enabled,
        settings.memory_enabled,
        settings.mcp_enabled,
        settings.hitl_enabled,
        settings.rate_limit_enabled,
    )
    settings.db_enabled = False
    settings.redis_enabled = False
    settings.memory_enabled = False
    settings.mcp_enabled = False
    settings.hitl_enabled = True
    settings.rate_limit_enabled = False
    reset_limiter()
    sid = f"hitl-{uuid.uuid4().hex[:8]}"
    path = str(ROOT / "app" / "sessions" / "server" / "hitl-test" / f"{sid}.json")
    agent = None
    try:
        with TestClient(app) as client:
            miss = client.post(
                "/api/confirm",
                json={"session_id": "no-such-hitl-session", "token": "x", "approved": True},
            )
            if miss.status_code != 404:
                _fail(f"无会话应 404，实际 {miss.status_code} {miss.text[:200]}")

            demo = client.get("/demo")
            if demo.status_code != 200 or "confirm_required" not in demo.text:
                _fail(f"GET /demo 应 200 且含确认流脚本，实际 {demo.status_code}")

            agent = YierAgent(session_path=path, user_id="hitl-u")
            tok = json.loads(
                agent.tool_manager.execute_tool("apply_refund", REASON)
            )["confirm_token"]
            registry._agents[sid] = agent
            registry._user_of[sid] = "hitl-u"

            bad = client.post(
                "/api/confirm",
                json={"session_id": sid, "token": "deadbeef", "approved": True},
            )
            if bad.status_code != 400:
                _fail(f"坏 token 应 400，实际 {bad.status_code} {bad.text[:200]}")

            ok = client.post(
                "/api/confirm",
                json={"session_id": sid, "token": tok, "approved": True},
            )
            if ok.status_code != 200:
                _fail(f"批准应 200，实际 {ok.status_code} {ok.text[:300]}")
            body = ok.json()
            if not body.get("executed") or not (body.get("result") or {}).get("success"):
                _fail(f"批准应当场执行退款，实际 {body}")
        _ok("confirm 404/400/200；demo 页可访问")
    finally:
        if sid in getattr(registry, "_agents", {}):
            registry.remove(sid)
        (
            settings.db_enabled,
            settings.redis_enabled,
            settings.memory_enabled,
            settings.mcp_enabled,
            settings.hitl_enabled,
            settings.rate_limit_enabled,
        ) = old
        reset_limiter()
        if agent is not None:
            try:
                agent.close()
            except Exception:
                pass


def test_shared_gate():
    print("\n[8/8] Multi-Agent 共用一把会话级闸")
    from app.multi_agent.orchestrator import MultiAgentOrchestrator

    old = (
        settings.db_enabled,
        settings.redis_enabled,
        settings.memory_enabled,
        settings.mcp_enabled,
        settings.hitl_enabled,
    )
    settings.db_enabled = False
    settings.redis_enabled = False
    settings.memory_enabled = False
    settings.mcp_enabled = False
    settings.hitl_enabled = True
    sid = uuid.uuid4().hex[:8]
    path = str(ROOT / "app" / "sessions" / "server" / "hitl-test" / f"{sid}.json")
    orch = None
    try:
        orch = MultiAgentOrchestrator(session_path=path, user_id="hitl-multi")
        postsale = orch.agents["postsale"].tool_manager
        presale = orch.agents["presale"].tool_manager
        if postsale.hitl is not orch.hitl or presale.hitl is not orch.hitl:
            _fail("子 Agent 必须共享 Orchestrator 的 ConfirmationGate")
        data = json.loads(postsale.execute_tool("apply_refund", REASON))
        if data.get("status") != "confirm_required":
            _fail(f"售后 apply_refund 应拦截，实际 {data}")
        orch.reset()
        if orch.hitl.pending or orch.hitl.allowed_orders:
            _fail("reset() 应清空 HITL 状态")
        _ok("多 Agent 共享闸门；reset 清空白名单")
    finally:
        (
            settings.db_enabled,
            settings.redis_enabled,
            settings.memory_enabled,
            settings.mcp_enabled,
            settings.hitl_enabled,
        ) = old
        if orch is not None:
            try:
                orch.close()
            except Exception:
                pass


def main():
    print("=" * 60)
    print("HITL 确认流")
    print("=" * 60)
    test_intercept_blocks()
    test_allow_then_other_order()
    test_deny()
    test_disabled_skips()
    test_utterance()
    test_resolve_on_agent()
    test_http_confirm()
    test_shared_gate()
    print("\n全部通过。")


if __name__ == "__main__":
    main()
