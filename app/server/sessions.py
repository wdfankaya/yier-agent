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
import time
from app.locks import KeyedLocks
from app.agent.maintenance import has_pending_work
from app.observability import failure, event
from pathlib import Path

from app.config.settings import settings
from app.server.identifiers import validate_id, safe_path

SESSION_DIR = Path("app/sessions/server")


def session_path_for(user_id: str, session_id: str) -> str:
    """生成会话存储路径；数据库模式下用作会话键。"""
    return str(safe_path(SESSION_DIR, validate_id(user_id), f"{validate_id(session_id)}.json"))


class SessionCapacityError(RuntimeError):
    pass


class SessionRegistry:
    """进程内常驻 Agent 的注册表。加锁保护 dict（FastAPI 线程池会并发访问）。"""

    def __init__(self) -> None:
        self._agents: dict[tuple[str, str], object] = {}
        self._busy = set()
        self._lock = threading.RLock()
        self._busy_lock = self._lock
        self._initializing = set()
        self._last_used = {}
        self._key_locks = KeyedLocks()
        self._cleanup_tasks = set()
        self._evicting = set()
        self._closing = False

    # ---------- 构造 ----------
    def _build_agent(self, session_id: str, user_id: str):
        """构造单/多 Agent（按 settings.multi_agent_enabled 切换）。"""
        if settings.workflow_enabled:
            from app.agent.workflow import GraphAgent
            return GraphAgent(session_path=session_path_for(user_id,session_id),user_id=user_id)
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
            key = self._key(session_id, user_id)
            self._busy.discard(key)
            if key in self._agents:
                self._last_used[key] = time.monotonic()

    def get(self, session_id: str, user_id: str):
        with self._lock:
            return self._agents.get(self._key(session_id, user_id))

    def open(self):
        with self._lock:
            self._closing = False

    def _obtain(self, session_id, user_id, restore_only=False):
        key = self._key(session_id, user_id)
        with self._key_locks.hold(key):
            with self._lock:
                if self._closing:
                    raise SessionCapacityError("Service is shutting down")
                agent = self._agents.get(key)
                if agent is not None:
                    self._last_used[key] = time.monotonic()
                    return agent
                if len(self._agents) + len(self._initializing) >= settings.session_max_resident:
                    raise SessionCapacityError("Resident session capacity reached")
                self._initializing.add(key)
            try:
                if restore_only:
                    if settings.db_enabled:
                        from app.server.repository import sync_load_session_state
                        exists = sync_load_session_state(user_id, session_id) is not None
                    else:
                        exists = Path(session_path_for(user_id, session_id)).is_file()
                    if not exists and settings.workflow_enabled:
                        from app.agent.workflow import checkpoint_store, thread_key
                        from app.agent.async_utils import run_sync
                        async def graph_exists():
                            async with checkpoint_store() as saver:
                                return await saver.aget_tuple({'configurable':{'thread_id':thread_key(user_id,session_path_for(user_id,session_id))}}) is not None
                        exists = run_sync(graph_exists())
                    if not exists:
                        return None
                agent = self._build_agent(session_id, user_id)
                with self._lock:
                    closing = self._closing
                    if not closing:
                        self._agents[key] = agent
                        self._last_used[key] = time.monotonic()
                if closing:
                    # Construction runs in a worker; this client has not entered the service loop.
                    from app.agent.maintenance import close_resources
                    asyncio.run(close_resources(agent))
                    raise SessionCapacityError("Service is shutting down")
                return agent
            finally:
                with self._lock:
                    self._initializing.discard(key)

    def get_or_create(self, session_id: str, user_id: str):
        return self._obtain(session_id, user_id)

    def get_or_restore(self, session_id: str, user_id: str):
        agent = self._obtain(session_id, user_id, restore_only=True)
        return (agent, user_id) if agent is not None else None

    def remove(self, session_id: str, user_id: str) -> bool:
        key = self._key(session_id, user_id)
        with self._lock:
            self._last_used.pop(key, None)
            return self._agents.pop(key, None) is not None

    async def reap_idle(self, force_room=False):
        now = time.monotonic()
        candidates = []
        with self._lock:
            pressure = force_room and len(self._agents) + len(self._initializing) >= settings.session_max_resident
            for key in sorted(self._agents, key=lambda k: self._last_used.get(k, now)):
                agent = self._agents[key]
                if key in self._busy or key in self._initializing or has_pending_work(agent):
                    continue
                gate = getattr(agent, "hitl", None)
                if gate is not None:
                    with gate._lock:
                        gate._expire()
                        if gate.pending:
                            continue
                expired = now - self._last_used.get(key, now) >= settings.session_idle_ttl
                if not expired and not pressure:
                    continue
                self._busy.add(key)
                self._evicting.add(key)
                candidates.append((key, agent))
                pressure = False
        if not candidates:
            return
        semaphore = asyncio.Semaphore(8)
        async def evict(key, agent):
            saved = False
            try:
                async with semaphore:
                    if hasattr(agent, "save"):
                        await asyncio.to_thread(agent.save)
                    saved = True
                    await agent.aclose()
                    event("session_evicted")
            except Exception as exc:
                failure("session_eviction_failed", exc)
            finally:
                with self._lock:
                    if saved and self._agents.get(key) is agent:
                        self._agents.pop(key, None)
                        self._last_used.pop(key, None)
                    self._busy.discard(key)
                    self._evicting.discard(key)
        tasks = {asyncio.create_task(evict(key, agent)) for key, agent in candidates}
        self._cleanup_tasks.update(tasks)
        for task in tasks:
            task.add_done_callback(self._cleanup_tasks.discard)
        _, pending = await asyncio.wait(tasks, timeout=settings.shutdown_timeout)
        for task in pending:
            task.cancel()

    async def close_all(self, timeout=None):
        """Close concurrently with one total budget, not N sequential budgets."""
        budget = settings.shutdown_timeout if timeout is None else timeout
        with self._lock:
            self._closing = True
            agents = [agent for key, agent in self._agents.items() if key not in self._evicting]
            self._agents.clear()
            self._last_used.clear()
        async def close(agent):
            try:
                await agent.aclose()
            except Exception as exc:
                failure("session_close_failed", exc)
        tasks = {asyncio.create_task(close(agent)) for agent in agents}
        tasks.update(self._cleanup_tasks)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=max(0, budget))
            for task in pending:
                task.cancel()
            if pending:
                event("session_close_timeout", count=len(pending))
                # Yield so cancellation can enter resource-release finally blocks.
                await asyncio.sleep(0)

    def __len__(self):
        with self._lock:
            return len(self._agents)
