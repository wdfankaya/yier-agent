"""技能加载工具：让 Agent 在 ReAct 循环中按需加载 Skill 指令。

与 memory_tool.py 相同——skill manager 不再放模块级全局，
改由 ToolManager 构造注入 + ContextVar 按调用上下文绑定。
服务端多 Agent 并存时各读各的 manager，不再串。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from app.agent.tools.decorator import tool

if TYPE_CHECKING:
    from app.agent.skills.loader import SkillManager

_skill_manager_ctx = ContextVar("skill_manager", default=None)


def get_current_skill_manager():
    """读取当前上下文中绑定的 skill manager（无则 None）。"""
    return _skill_manager_ctx.get()


def set_skill_manager(manager) -> None:
    """设置当前上下文的 skill manager（测试/独立调用用）。"""
    _skill_manager_ctx.set(manager)


@contextmanager
def bind_skill_manager(manager):
    """以指定 manager 进入一段上下文，退出自动还原（ToolManager 调用时用）。"""
    token = _skill_manager_ctx.set(manager)
    try:
        yield
    finally:
        _skill_manager_ctx.reset(token)


@tool(
    desc=(
        "加载指定技能的完整操作指令。"
        "当用户问题匹配某个可用技能时，调用此工具获取该技能的详细处理流程，"
        "然后按流程指引使用已有工具完成用户请求。"
        "可用技能会在系统提示中列出。"
    ),
    params={
        "skill_name": "要加载的技能名称，如 process-return、track-order、product-recommend"
    },
)
def load_skill(skill_name: str) -> dict:
    """加载指定技能的完整指令。Agent 调用后按指令处理用户问题。"""
    manager = get_current_skill_manager()
    if manager is None:
        return {"success": False, "error": "技能系统未启用"}
    return manager.load_skill(skill_name)
