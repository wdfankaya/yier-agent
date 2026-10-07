"""工具注册表：执行分发入口。

工具函数在各自文件里用 @tool(desc=..., params=...) 装饰，import 时自动完成双登记：
  - decorator._TOOL_MAP        执行表（名字 → 函数）
  - decorator.TOOL_DEFINITIONS OpenAI schema（给 LLM 看）

本模块职责：
  1. import 各工具模块，触发装饰器注册（import 顺序 = 工具定义展示顺序）
  2. 透出 decorator 产出的两份登记（TOOL_DEFINITIONS / _TOOL_MAP）
  3. 保留 execute_tool 分发执行
"""

import json

# 触发各工具模块的 @tool 注册（import 顺序即 TOOL_DEFINITIONS 展示顺序）
import app.agent.tools.order  # noqa: F401
import app.agent.tools.product  # noqa: F401
import app.agent.tools.logistics  # noqa: F401
import app.agent.tools.knowledge  # noqa: F401
import app.agent.tools.user_orders  # noqa: F401
import app.agent.tools.refund  # noqa: F401
import app.agent.tools.memory_tool  # noqa: F401
import app.agent.tools.skill_tool  # noqa: F401

from app.agent.tools.decorator import TOOL_DEFINITIONS, _TOOL_MAP  # noqa: E402


def execute_tool(name: str, arguments: dict) -> str:
    """根据工具名称分发执行，返回 JSON 字符串结果。"""
    func = _TOOL_MAP.get(name)
    if not func:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        result = func(**arguments)
    except Exception as e:
        result = {"error": f"工具执行出错: {e}"}
    return json.dumps(result, ensure_ascii=False)
