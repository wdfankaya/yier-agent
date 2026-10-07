"""会话注册表：每 session 进程内常驻 Agent（热态）的 dict。

状态模型：
- 进程内 dict 只是「热态副本」，请求命中常驻 Agent 时不回源
- 真相源在 PG（消息级落库）
- Redis 只加速冷读：dict 未命中、进程重启后 load_session 先查缓存再回源 PG
- 单 worker 部署；多 worker 需「每请求重建无状态 Agent + PG 回放」，当前未实现
- session_path 按 user_id/session_id 隔离：app/sessions/server/{user_id}/{session_id}.json
  （数据库模式下文件路径用于生成 session_key，真实数据在 PG；JSON 是 db_enabled=false 的回退）
- 同一 user_id 的不同会话共享 LTM；不同 user_id 的 LTM 隔离（注入生效）

接口对齐 YierAgent / MultiAgentOrchestrator：chat() / reset() / close() / raw_messages。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from app.config.settings import settings

SESSION_DIR = Path("app/sessions/server")


def session_path_for(user_id: str, session_id: str) -> str:
    """生成会话存储路径；数据库模式下用作会话键。"""
    return str(SESSION_DIR / user_id / f"{session_id}.json")


class SessionRegistry:
    """进程内常驻 Agent 的注册表。加锁保护 dict（FastAPI 线程池会并发访问）。"""

    def __init__(self) -> None:
        self._agents: dict[str, object] = {}
        self._user_of: dict[str, str] = {}
        self._lock = threading.RLock()

    # ---------- 构造 ----------
    def _build_agent(self, session_id: str, user_id: str):
        """构造单/多 Agent（按 settings.multi_agent_enabled 切换）。"""
        if settings.multi_agent_enabled:
            from app.multi_agent.orchestrator import MultiAgentOrchestrator
            return MultiAgentOrchestrator(
                session_path=session_path_for(user_id, session_id),
                user_id=user_id,
            )
        from app.agent.chat import YierAgent
        return YierAgent(
            session_path=session_path_for(user_id, session_id),
            user_id=user_id,
        )

    # ---------- 读写 ----------
    def get(self, session_id: str):
        """返回常驻 agent；不存在返回 None（不自动构造）。"""
        with self._lock:
            return self._agents.get(session_id)

    def get_or_create(self, session_id: str, user_id: str):
        """会话存在则返回常驻热态；否则按其历史（Redis 冷缓存 / PG / JSON）构造并驻留。"""
        with self._lock:
            agent = self._agents.get(session_id)
            if agent is not None:
                return agent
            agent = self._build_agent(session_id, user_id)
            self._agents[session_id] = agent
            self._user_of[session_id] = user_id
            return agent

    def _restore_from_disk(self, session_id: str):
        """服务重启后：找回该会话的归属用户并重建驻留。

        DB 模式：GET/DELETE 只带 session_id、没带 user_id，
        到 PG 里查最近活跃的该 session_key 归属谁，再按 (user, key) 重建。
        重建时 Agent.__init__ → load_session：先查 Redis，未命中再回源 PG。
        回退模式：按 JSON 文件布局反推用户。
        """
        if settings.db_enabled:
            from app.server.repository import sync_find_session_owner
            user_id = sync_find_session_owner(session_id)
            if user_id is None:
                return None
            return self.get_or_create(session_id, user_id)

        hits = list(SESSION_DIR.glob(f"*/{session_id}.json"))
        if not hits:
            return None
        user_id = hits[0].parent.name
        return self.get_or_create(session_id, user_id)

    def get_or_restore(self, session_id: str):
        """取常驻 agent；无则尝试从磁盘恢复。返回 (agent, user_id) 或 None。"""
        with self._lock:
            agent = self._agents.get(session_id)
            if agent is not None:
                return agent, self._user_of.get(session_id, "default")
        restored = self._restore_from_disk(session_id)
        if restored is None:
            return None
        return restored, self._user_of.get(session_id, "default")

    def remove(self, session_id: str) -> bool:
        """摘除常驻 agent（不删历史文件）。"""
        with self._lock:
            if session_id not in self._agents:
                return False
            del self._agents[session_id]
            self._user_of.pop(session_id, None)
            return True

    # ---------- 生命周期 ----------
    async def close_all(self, timeout: float = 5.0) -> None:
        """优雅关闭全部常驻 Agent（await aclose）。

        aclose() 会 await 后台 LTM 任务再巩固一次。单个会话 5s 预算，超时放弃
        （kill -9 走不到这里；平时靠每 N 轮定期巩固兜底最后窗口）。
        """
        with self._lock:
            agents = list(self._agents.values())
            self._agents.clear()
            self._user_of.clear()

        for agent in agents:
            closer = getattr(agent, "aclose", None)
            try:
                if closer is not None:
                    await asyncio.wait_for(closer(), timeout=timeout)
                else:
                    await asyncio.wait_for(
                        asyncio.to_thread(agent.close),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError:  # noqa: PERF203 —— 单失败不中断整体
                print(f"⏱️  会话 close 超时（>{timeout}s），放弃本轮巩固")
            except Exception as e:  # noqa: BLE001 —— 单失败不中断整体
                print(f"⚠️  会话 close 失败: {e}")

    def __len__(self) -> int:
        return len(self._agents)
