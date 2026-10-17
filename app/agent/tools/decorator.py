"""One Pydantic argument model drives both runtime validation and tool schema."""
import inspect
import json
import re
from functools import wraps
from typing import Callable, Optional, get_type_hints
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

_TOOL_MAP: dict[str, Callable] = {}
_TOOL_MODELS: dict[str, type[BaseModel]] = {}
TOOL_DEFINITIONS: list[dict] = []

class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

class ToolInputError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("Invalid tool arguments")

    def payload(self, name):
        return {"success": False, "error": "工具参数校验失败，请按字段要求修正后重试",
                "code": "invalid_tool_arguments", "tool": name, "issues": self.issues}

def parse_arguments(arguments):
    if isinstance(arguments, str):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate field")
                result[key] = value
            return result
        def invalid_constant(value):
            raise ValueError("Non-finite number")
        try:
            arguments = json.loads(arguments, object_pairs_hook=pairs, parse_constant=invalid_constant)
        except (ValueError, RecursionError):
            raise ToolInputError([{"field": [], "type": "json_invalid", "message": "参数必须是合法 JSON，不能含重复字段或非有限数值"}]) from None
    if not isinstance(arguments, dict):
        raise ToolInputError([{"field": [], "type": "object_required", "message": "参数必须是 JSON 对象"}])
    return arguments

def validate_arguments(name, arguments):
    data = parse_arguments(arguments)
    try:
        return _TOOL_MODELS[name].model_validate(data).model_dump()
    except ValidationError as exc:
        raise ToolInputError([{"field": list(e["loc"]), "type": e["type"], "message": e["msg"]}
                              for e in exc.errors(include_input=False, include_context=False, include_url=False)]) from None

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
    """Register a validated function; annotations may include Pydantic constraints."""
    def decorator(func: Callable) -> Callable:
        name = func.__name__
        signature = inspect.signature(func)
        hints = get_type_hints(func, include_extras=True)
        descriptions = _docstring_param_descs(func.__doc__)
        descriptions.update(params or {})
        fields = {}
        for pname, parameter in signature.parameters.items():
            if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                raise TypeError("Tools require named parameters")
            if pname not in hints:
                raise TypeError(f"Tool parameter {pname} requires a type annotation")
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[pname] = (hints[pname], Field(default=default, description=descriptions.get(pname)))
        model = create_model(f"{name}Arguments", __base__=ToolArguments, **fields)
        _TOOL_MODELS[name] = model

        @wraps(func)
        def validated(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            return func(**validate_arguments(name, bound.arguments))

        _TOOL_MAP[name] = validated
        definition = {"type": "function", "function": {
            "name": name, "description": desc, "parameters": model.model_json_schema(),
        }}
        # Reloading a module must not publish duplicate tool names.
        TOOL_DEFINITIONS[:] = [d for d in TOOL_DEFINITIONS if d["function"]["name"] != name]
        TOOL_DEFINITIONS.append(definition)
        return validated
    return decorator
