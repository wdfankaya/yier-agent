"""LLM 超时 + 指数退避重试。

只对 429 / 超时重试；4xx 参数错误、鉴权失败立刻抛。
退避 1s / 2s / 4s（`llm_retry_base * 2^i`），最多再试 llm_max_retries 次。
每次调用都用新的 awaitable（失败的 coroutine 不能重用）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from app.config.settings import settings

T = TypeVar("T")


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    return type(exc).__name__ in {"RateLimitError", "APITimeoutError"}


async def with_timeout_retry(
    factory: Callable[[], Awaitable[T]],
    *,
    timeout: float | None = None,
    retries: int | None = None,
    base: float | None = None,
) -> T:
    """factory() 每次尝试都要返回新的 awaitable。"""
    timeout = settings.llm_timeout if timeout is None else timeout
    retries = settings.llm_max_retries if retries is None else retries
    base = settings.llm_retry_base if base is None else base
    last: BaseException | None = None
    attempts = retries + 1
    for i in range(attempts):
        try:
            return await asyncio.wait_for(factory(), timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            if not is_retryable(e) or i >= retries:
                raise
            delay = base * (2 ** i)
            print(
                f"⚠️  LLM {type(e).__name__}，{delay:g}s 后重试 "
                f"({i + 1}/{retries})"
            )
            await asyncio.sleep(delay)
    assert last is not None
    raise last


async def llm_create(client: Any, **kwargs):
    return await with_timeout_retry(
        lambda: client.chat.completions.create(**kwargs),
    )


async def llm_parse(client: Any, **kwargs):
    return await with_timeout_retry(
        lambda: client.beta.chat.completions.parse(**kwargs),
    )
