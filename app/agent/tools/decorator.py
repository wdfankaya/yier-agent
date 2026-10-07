"""工具装饰器：从函数签名自动生成 OpenAI function schema，消灭手写双登记。

在工具函数上写 ``@tool(desc=..., params={...})``，import 时自动完成：
  ① 把 {函数名: 函数} 登记进 _TOOL_MAP（execute_tool 分发执行用）
  ② 用 inspect.signature 解析参数注解/默认值 → JSON Schema properties / required
  ③ 参数级 description 来自 params={参数名: 描述}，缺省时回退到函数 docstring

规则：
  - 无默认值的参数 → required；有默认值的参数 → 可选，并把 default 写进 schema
  - 类型注解映射 int→integer / float→number / bool→boolean / Optional[X]→X，其余 string
  - desc 是模型决定"何时调用/填什么"的关键语义，务必写全；删登记样板但别删语义
"""

import inspect
import re
from typing import Callable, Optional

_TOOL_MAP: dict[str, Callable] = {}
TOOL_DEFINITIONS: list[dict] = []


def _json_type(annotation) -> str:
    """把类型注解映射成 JSON Schema 的 type 字符串。"""
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    # Optional[X] / X | None → 取非 None 的基础类型
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
    return "string"


def _docstring_param_descs(doc: Optional[str]) -> dict[str, str]:
    """从 docstring 里粗提取 '参数名: 描述' 行，作为 params 缺省时的回退。"""
    out: dict[str, str] = {}
    if not doc:
        return out
    for line in doc.splitlines():
        m = re.match(r"^\s*(\w+)\s*:\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def tool(desc: str, params: Optional[dict[str, str]] = None):
    """装饰器：import 时注册函数与其自动生成的 schema。

    Args:
        desc: 工具的整体 description（模型判断何时调用的依据）。
        params: {参数名: 该参数的 description}，用于补全参数级语义。
    """
    def decorator(func: Callable) -> Callable:
        name = func.__name__
        _TOOL_MAP[name] = func

        doc_params = _docstring_param_descs(func.__doc__)
        properties: dict = {}
        required: list[str] = []

        for pname, p in inspect.signature(func).parameters.items():
            if pname in ("self", "cls"):
                continue
            prop: dict = {"type": _json_type(p.annotation)}

            pdesc = ""
            if params and pname in params:
                pdesc = params[pname]
            elif pname in doc_params:
                pdesc = doc_params[pname]
            if pdesc:
                prop["description"] = pdesc

            if p.default is not inspect.Parameter.empty:
                prop["default"] = p.default
            else:
                required.append(pname)
            properties[pname] = prop

        TOOL_DEFINITIONS.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
        return func

    return decorator
