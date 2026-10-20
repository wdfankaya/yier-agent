"""记忆查询工具：让 Agent 在 ReAct 循环中主动查询用户记忆。

manager 不再放模块级全局（多 Agent 并存时后初始化的会覆盖先前的，
导致 recall_user_memory 串记忆）。改为按「调用上下文」注入：
  - 服务端：ToolManager 构造时持有本 agent 的 memory_manager，执行工具时 bind
  - 测试/独立调用：可调 set_memory_manager() 设置当前上下文的 manager

ContextVar 本质是「随调用链走的变量」：同一时刻不同 Agent/线程各读各的，
不存在模块级覆盖问题。但绑定隔离不会复制 manager，也不负责共享状态的并发控制。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal

from app.agent.tools.decorator import tool
from app.agent.tools.arguments import MemoryQuery, TopK

if TYPE_CHECKING:
    from app.agent.memory.manager import MemoryManager

_memory_manager_ctx = ContextVar("memory_manager", default=None)


def get_current_memory_manager():
    """读取当前上下文中绑定的 memory manager（无则 None）。"""
    return _memory_manager_ctx.get()


def set_memory_manager(manager) -> None:
    """设置当前上下文的 memory manager（测试/独立调用用）。"""
    _memory_manager_ctx.set(manager)


@contextmanager
def bind_memory_manager(manager):
    """以指定 manager 进入一段上下文，退出自动还原（ToolManager 调用时用）。"""
    token = _memory_manager_ctx.set(manager)
    try:
        yield
    finally:
        _memory_manager_ctx.reset(token)


@tool(
    desc=(
        "查询当前用户的记忆信息，包括本次对话提取的短期记忆和跨会话的长期记忆。"
        "当需要回顾用户的偏好、历史问题、会员信息等时使用。"
    ),
    params={"query": "可选的查询关键词，用于过滤记忆内容", "category": "可选分类", "top_k": "每类返回条数，1到5"},
)
def recall_user_memory(query: MemoryQuery = "", category: Literal["preference", "identity", "issue", "other"] | None = None, top_k: TopK = 5) -> dict:
    """查询当前用户的记忆信息（长期记忆和短期记忆）。"""
    manager = get_current_memory_manager()
    if manager is None or not manager.memory_enabled:
        return {"success": False, "error": "记忆系统未启用"}

    from dataclasses import asdict
    from app.agent.context_summary import search_sources
    long_term = manager.ltm.recall(query, category, top_k)
    manager.stm.reconcile(manager.ltm.inspect(), manager.ltm.reset_at)
    return {
        "success": True,
        "short_term_facts": manager.stm.recall(query, category, top_k),
        "long_term_facts": [asdict(f) for f in long_term],
        "conversation_evidence": search_sources(manager.ltm.user_id, manager.source_session, query, top_k),
        "notice": "仅返回相关且有效的来源证据。订单、退款和会员状态必须调用业务工具重新查询。",
    }
