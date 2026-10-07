"""同步/异步边界小工具。

服务端跑在 uvicorn 的长寿命事件循环上，用 await；CLI / 脚本式测试没有循环，
用 run_sync（内部 asyncio.run）把协程跑完。禁止在已经在跑的 loop 里调 run_sync
（会直接 RuntimeError）——那种路径必须 await 对应的 a* 方法。
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[object, object, T]) -> T:
    """在【没有运行中事件循环】的线程里跑完一个协程（CLI / tests / sandbox.run）。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "run_sync() 不能在运行中的事件循环里调用，请改 await 对应的 async API"
    )
