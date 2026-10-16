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
from app.server.identifiers import validate_id, safe_path

SESSION_DIR = Path("app/sessions/server")


def session_path_for(user_id: str, session_id: str) -> str:
    """生成会话存储路径；数据库模式下用作会话键。"""
    return str(safe_path(SESSION_DIR, validate_id(user_id), f"{validate_id(session_id)}.json"))


class SessionRegistry:
    """进程内常驻 Agent 的注册表。加锁保护 dict（FastAPI 线程池会并发访问）。"""

    def __init__(self) -> None:
        self._agents: dict[tuple[str, str], object] = {}
        self._busy: set[tuple[str, str]] = set()
        self._busy_lock = threading.Lock()
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

    def _key(self, session_id: str, user_id: str):
        return validate_id(user_id), validate_id(session_id)

    def acquire(self, session_id: str, user_id: str) -> bool:
        key = self._key(session_id, user_id)
        with self._busy_lock:
            if key in self._busy:
                return False
            self._busy.add(key)
            return True

    def release(self, session_id: str, user_id: str) -> None:
        with self._busy_lock:
            self._busy.discard(self._key(session_id, user_id))

    def get(self, session_id: str, user_id: str):
        with self._lock:
            return self._agents.get(self._key(session_id, user_id))

    def get_or_create(self, session_id: str, user_id: str):
        key = self._key(session_id, user_id)
        with self._lock:
            agent = self._agents.get(key)
            if agent is None:
                agent = self._build_agent(session_id, user_id)
                self._agents[key] = agent
            return agent

    def get_or_restore(self, session_id: str, user_id: str):
        key = self._key(session_id, user_id)
        with self._lock:
            agent = self._agents.get(key)
            if agent is not None:
                return agent, user_id
            if settings.db_enabled:
                from app.server.repository import sync_load_session_state
                exists = sync_load_session_state(user_id, session_id) is not None
            else:
                exists = Path(session_path_for(user_id, session_id)).is_file()
            if not exists:
                return None
            return self.get_or_create(session_id, user_id), user_id

    def remove(self, session_id: str, user_id: str) -> bool:
        with self._lock:
            return self._agents.pop(self._key(session_id, user_id), None) is not None

    # ---------- 生命周期 ----------
    async def close_all(self, timeout: float = 5.0) -> None:
        """优雅关闭全部常驻 Agent（await aclose）。

        aclose() 会 await 后台 LTM 任务再巩固一次。单个会话 5s 预算，超时放弃
        （kill -9 走不到这里；平时靠每 N 轮定期巩固兜底最后窗口）。
        """
        with self._lock:
            agents = list(self._agents.values())
            self._agents.clear()

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
