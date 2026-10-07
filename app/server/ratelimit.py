"""令牌桶限流（自实现）。

按 user_id 限流（默认 10 次/分钟，突发容量 = 10）。
- 单机：进程内 dict + asyncio.Lock（当前单 worker 部署够用）
- Redis 可用时走 Lua：HGET/计算/HSET/EXPIRE 一次 EVAL，多实例也原子
- Redis 挂了回落到内存桶（单进程仍限流；多 worker 会各算各的——见 README）

令牌桶通过 Redis Lua 原子更新；Redis 不可用时回退到进程内限流。
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

# KEYS[1]=bucket key
# ARGV: rate, capacity, requested, ttl
# 用 Redis TIME，避免多机墙钟不准。
_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
end
local delta = now - ts
if delta < 0 then delta = 0 end
tokens = math.min(capacity, tokens + delta * rate)
local allowed = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)
return allowed
"""


class TokenBucket:
    """rate=每秒补多少令牌；capacity=桶深（突发上限）。"""

    def __init__(self, rate: float, capacity: int, enabled: bool = True) -> None:
        self.rate = rate
        self.capacity = max(0, int(capacity))
        self.enabled = enabled
        self._local: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls) -> TokenBucket:
        from app.config.settings import settings
        per_min = max(0, int(settings.rate_limit_per_minute))
        burst = int(settings.rate_limit_burst)
        if burst <= 0:
            burst = per_min
        rate = (per_min / 60.0) if per_min else 0.0
        return cls(rate=rate, capacity=burst, enabled=bool(settings.rate_limit_enabled))

    async def acquire(self, key: str, tokens: float = 1.0) -> bool:
        """拿到令牌返回 True；超限 False。enabled=False 时永远放行。"""
        if not self.enabled:
            return True
        allowed = await asyncio.to_thread(self._try_redis, key, tokens)
        if allowed is not None:
            return allowed
        return await self._acquire_local(key, tokens)

    def _try_redis(self, key: str, tokens: float) -> Optional[bool]:
        from app.server.cache import get_cache
        ttl = 120
        if self.rate > 0:
            ttl = max(120, int(2 * self.capacity / self.rate) + 1)
        out = get_cache().eval_lua(
            _BUCKET_LUA,
            [f"bucket:{key}"],
            [self.rate, self.capacity, tokens, ttl],
        )
        if out is None:
            return None
        return bool(int(out))

    async def _acquire_local(self, key: str, requested: float) -> bool:
        async with self._lock:
            now = time.monotonic()
            tokens, ts = self._local.get(key, (float(self.capacity), now))
            delta = max(0.0, now - ts)
            tokens = min(float(self.capacity), tokens + delta * self.rate)
            if tokens >= requested:
                self._local[key] = (tokens - requested, now)
                return True
            self._local[key] = (tokens, now)
            return False


_limiter: Optional[TokenBucket] = None


def get_limiter() -> TokenBucket:
    global _limiter
    if _limiter is None:
        _limiter = TokenBucket.from_settings()
    return _limiter


def reset_limiter(instance: Optional[TokenBucket] = None) -> None:
    """测试用：注入指定桶；None 表示下次按 settings 重建。"""
    global _limiter
    _limiter = instance
