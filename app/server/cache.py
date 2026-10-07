"""Redis 冷读缓存。

状态模型：
- 热态：进程内 SessionRegistry dict（本轮请求直接用 Agent，不碰 Redis）
- 真相源：PostgreSQL（每轮 save_session 先写 PG）
- Redis：只做「进程重启后从存储重建会话」的加速层

缓存策略 = Cache-Aside 读 + 写穿：
- 读：GET session:{user}:{key} 命中 → 直接用；未命中 → 查 PG → SET 回填（TTL 30min）
- 写：先 INSERT/整窗覆盖 PG，成功后再 SET 缓存副本
  （计划里的 RPUSH 对应「追加消息」；本项目 save_session 是整窗覆盖——压缩会删旧消息——
   所以缓存也整包 SET，避免 list 里残留已被压缩掉的条目）

硬约束：Redis 挂了服务不挂。所有命令短超时，失败当 miss / 写缓存失败忽略。

incr_window 使用 INCR + EXPIRE NX 提供固定窗口计数。
"""

from __future__ import annotations

import json
from typing import Optional

# 短超时：缓存是加速层，连不上就立刻回源，绝不为 Redis 多等几秒。
_SOCKET_TIMEOUT = 0.3

SESSION_KEY_PREFIX = "session:"
RATE_KEY_PREFIX = "rate:"


class SessionCache:
    """会话快照缓存 + 窗口计数。enabled=False 时全部方法空操作。"""

    def __init__(
        self,
        enabled: bool,
        url: str = "redis://127.0.0.1:6379/0",
        ttl: int = 1800,
        client=None,
    ) -> None:
        self.enabled = enabled
        self.url = url
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self._r = client
        self._warned = False

    @classmethod
    def from_settings(cls) -> SessionCache:
        from app.config.settings import settings
        return cls(
            enabled=bool(settings.redis_enabled),
            url=settings.redis_url,
            ttl=int(settings.session_cache_ttl),
        )

    # ---------- 连接 ----------
    def _client(self):
        if not self.enabled:
            return None
        if self._r is not None:
            return self._r
        try:
            import redis
            self._r = redis.Redis.from_url(
                self.url,
                decode_responses=True,
                socket_connect_timeout=_SOCKET_TIMEOUT,
                socket_timeout=_SOCKET_TIMEOUT,
            )
            return self._r
        except Exception as e:  # noqa: BLE001 —— 连不上就当缓存不存在
            self._fail("connect", e)
            return None

    def _fail(self, op: str, e: BaseException) -> None:
        self.errors += 1
        if not self._warned:
            print(
                f"⚠️  Redis {op} 失败，回源 PG（缓存挂了服务不挂）: "
                f"{type(e).__name__}: {e}"
            )
            self._warned = True

    def _recovered(self) -> None:
        if self._warned:
            print("🟢 Redis 已恢复")
            self._warned = False

    def status(self) -> str:
        """给 /health 用：ok / down / disabled。"""
        if not self.enabled:
            return "disabled"
        try:
            c = self._client()
            if c is None:
                return "down"
            c.ping()
            self._recovered()
            return "ok"
        except Exception:
            return "down"

    def close(self) -> None:
        if self._r is None:
            return
        try:
            self._r.close()
        except Exception:  # noqa: BLE001
            pass
        self._r = None

    # ---------- 会话快照 ----------
    @staticmethod
    def session_key(user_id: str, session_key: str) -> str:
        # session_key 跨用户不唯一（repository.find_session_owner 也是这个前提），必须带 user_id。
        return f"{SESSION_KEY_PREFIX}{user_id}:{session_key}"

    def get_session(self, user_id: str, session_key: str) -> Optional[dict]:
        """Cache-Aside 读。命中返回 {summary, messages, short_term_memory}；否则 None。"""
        c = self._client()
        if c is None:
            return None
        try:
            raw = c.get(self.session_key(user_id, session_key))
            self._recovered()
        except Exception as e:  # noqa: BLE001
            self._fail("get", e)
            return None
        if not raw:
            self.misses += 1
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self.misses += 1
            return None
        if not isinstance(data, dict) or "messages" not in data:
            self.misses += 1
            return None
        self.hits += 1
        return data

    def set_session(self, user_id: str, session_key: str, payload: dict) -> bool:
        """写穿的后半段：PG 已成功后更新缓存副本。失败返回 False，调用方忽略。"""
        c = self._client()
        if c is None:
            return False
        try:
            c.set(
                self.session_key(user_id, session_key),
                json.dumps(payload, ensure_ascii=False),
                ex=self.ttl,
            )
            self._recovered()
            return True
        except Exception as e:  # noqa: BLE001
            self._fail("set", e)
            return False

    def delete_session(self, user_id: str, session_key: str) -> None:
        c = self._client()
        if c is None:
            return
        try:
            c.delete(self.session_key(user_id, session_key))
            self._recovered()
        except Exception as e:  # noqa: BLE001
            self._fail("delete", e)

    # ---------- 固定窗口计数 ----------
    def incr_window(self, name: str, window_seconds: int = 60) -> Optional[int]:
        """INCR + EXPIRE NX。返回当前计数；Redis 不可用返回 None（限流应 fail-open）。

        原子性靠 pipeline：第一条 INCR 把计数 +1，EXPIRE NX 只在 key 新生时设窗口，
        避免每次请求把 TTL 刷新成滑动窗口。令牌桶使用独立的 Redis Lua 实现。
        """
        c = self._client()
        if c is None:
            return None
        try:
            pipe = c.pipeline()
            pipe.incr(name)
            pipe.expire(name, window_seconds, nx=True)
            n, _ = pipe.execute()
            self._recovered()
            return int(n)
        except Exception as e:  # noqa: BLE001
            self._fail("incr", e)
            return None

    def incr_user_rate(self, user_id: str, window_seconds: int = 60) -> Optional[int]:
        return self.incr_window(f"{RATE_KEY_PREFIX}{user_id}", window_seconds)

    def eval_lua(self, script: str, keys: list[str], args: list) -> Optional[object]:
        """跑一段 Lua。失败/未启用返回 None，调用方回落本地逻辑。"""
        c = self._client()
        if c is None:
            return None
        try:
            out = c.eval(script, len(keys), *keys, *args)
            self._recovered()
            return out
        except Exception as e:  # noqa: BLE001
            self._fail("eval", e)
            return None


_cache: Optional[SessionCache] = None


def get_cache() -> SessionCache:
    """进程内单例。lifespan / storage 都走这里，方便测试替换。"""
    global _cache
    if _cache is None:
        _cache = SessionCache.from_settings()
    return _cache


def reset_cache(instance: Optional[SessionCache] = None) -> None:
    """测试用：关掉旧连接并换成指定实例（None = 下次 get_cache 按 settings 重建）。"""
    global _cache
    if _cache is not None:
        _cache.close()
    _cache = instance


def aclose_cache() -> None:
    """lifespan shutdown：释放 Redis 连接。同步即可（close 是本地句柄，无网络往返必要）。"""
    reset_cache(None)
