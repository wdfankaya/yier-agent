"""ToolManager：统一管理本地工具和 MCP 工具。

当 MCP 启用时，通过 Streamable HTTP 连接 MCP Server 获取工具；
当 MCP 未启用或连接失败时，退回本地工具。

memory_manager / skill_manager 由构造方注入（按 Agent 持有），
执行本地工具时用 ContextVar 绑定到当前调用上下文，替换旧的模块级全局注入，
避免多 Agent 并存时串记忆/串技能。
"""

import json
import asyncio
from typing import Optional
from app.observability import stage, failure, event

from app.agent.tools.registry import TOOL_DEFINITIONS as LOCAL_TOOL_DEFINITIONS
from app.agent.tools.registry import execute_tool as local_execute_tool
from app.agent.tools.decorator import validate_arguments, parse_arguments, ToolInputError
from app.agent.tools.memory_tool import bind_memory_manager
from app.agent.tools.skill_tool import bind_skill_manager


class ToolManager:
    """聚合本地工具和 MCP 工具，提供统一的工具定义和调度接口。

    Args:
        memory_manager: 本 Agent 的 MemoryManager（可为 None，此时 recall_user_memory
            返回「记忆系统未启用」，等价于旧版未注入）。
        skill_manager:  本 Agent 的 SkillManager（可为 None，此时 load_skill 未启用）。
    """

    # 敏感操作执行前走 ConfirmationGate（精确参数、单次授权）。
    SENSITIVE_TOOLS: set[str] = {"apply_refund"}

    def is_sensitive(self, name: str) -> bool:
        """该工具是否属于需要用户确认的敏感操作。"""
        return name in self.SENSITIVE_TOOLS

    def __init__(
        self,
        use_mcp: bool = False,
        mcp_server_url: str = "",
        allowed_tools: Optional[set] = None,
        memory_manager=None,
        skill_manager=None,
        hitl=None,
    ):
        self._inflight: set[asyncio.Task] = set()
        self._mcp_client = None
        self._tool_source: dict[str, str] = {}
        self._tool_defs: list[dict] = []
        self.memory_manager = memory_manager
        self.skill_manager = skill_manager
        self.hitl = hitl

        if use_mcp and mcp_server_url:
            self._init_mcp(mcp_server_url)
        else:
            self._init_local()

        if allowed_tools is not None:
            self._filter_tools(allowed_tools)

    def _init_local(self):
        """只加载本地工具。"""
        self._tool_defs = list(LOCAL_TOOL_DEFINITIONS)
        for td in self._tool_defs:
            self._tool_source[td["function"]["name"]] = "local"

    def _init_mcp(self, server_url: str):
        """连接 MCP Server 加载工具；失败时降级到本地工具。"""
        from app.mcp_client import MCPClient

        try:
            self._mcp_client = MCPClient(server_url)
            mcp_tools = self._mcp_client.connect()
            print(f"🔗 [MCP] 已连接 {server_url}，发现 {len(mcp_tools)} 个工具")

            mcp_names = set()
            for td in mcp_tools:
                name = td["function"]["name"]
                mcp_names.add(name)
                self._tool_source[name] = "mcp"
            self._tool_defs = list(mcp_tools)

            for td in LOCAL_TOOL_DEFINITIONS:
                name = td["function"]["name"]
                if name not in mcp_names:
                    self._tool_defs.append(td)
                    self._tool_source[name] = "local"

        except Exception as e:
            print(f"⚠️  [MCP] 连接失败 ({e})，降级使用本地工具")
            if self._mcp_client:
                self._mcp_client.close()
                self._mcp_client = None
            self._init_local()

    def _filter_tools(self, allowed: set):
        """只保留白名单中的工具，用于子 Agent 工具隔离。"""
        self._tool_defs = [
            d for d in self._tool_defs
            if d["function"]["name"] in allowed
        ]
        self._tool_source = {
            k: v for k, v in self._tool_source.items()
            if k in allowed
        }

    @property
    def tool_definitions(self) -> list[dict]:
        return self._tool_defs

    def execute_tool(self, name: str, arguments: dict) -> str:
        """根据工具来源分发调用。敏感工具先过 HITL 闸，未授权不执行。"""
        source = self._tool_source.get(name)
        if source is None:
            return json.dumps({"error": "未知工具或当前 Agent 无权调用", "code": "unknown_tool"}, ensure_ascii=False)
        try:
            arguments = validate_arguments(name, arguments) if source == "local" else parse_arguments(arguments)
        except ToolInputError as exc:
            return json.dumps(exc.payload(name), ensure_ascii=False)
        if self.hitl is not None and self.is_sensitive(name):
            blocked = self.hitl.intercept(name, arguments or {})
            if blocked is not None:
                return json.dumps(blocked, ensure_ascii=False)

        if source == "mcp" and self._mcp_client:
            raw = self._mcp_client.call_tool(name, arguments)
            self._record_result(name, arguments, raw)
            return raw

        if source == "local":
            # 把本 Agent 的 memory/skill manager 绑定到调用上下文，
            # recall_user_memory / load_skill 才能拿到「本会话用户」的实例。
            with bind_memory_manager(self.memory_manager), \
                 bind_skill_manager(self.skill_manager):
                raw = local_execute_tool(name, arguments)
                self._record_result(name, arguments, raw)
                return raw

        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    def _record_result(self, name, arguments, raw):
        if self.hitl is not None and self.is_sensitive(name):
            try:
                result = json.loads(raw)
            except (ValueError, TypeError):
                result = {"raw": raw}
            self.hitl.record(name, arguments, result)

    @property
    def has_pending_work(self):
        return any(not task.done() for task in self._inflight)

    async def aexecute_tool(self, name: str, arguments: dict) -> str:
        """Bound caller waiting; keep late effects tracked and never auto-retry writes."""
        from app.config.settings import settings
        task = asyncio.create_task(asyncio.to_thread(self.execute_tool, name, arguments))
        self._inflight.add(task)
        def completed(t):
            self._inflight.discard(t)
            if not t.cancelled() and t.exception() is not None:
                failure("tool_background_failed", t.exception())
        task.add_done_callback(completed)
        with stage("tool"):
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=settings.tool_timeout)
            except asyncio.TimeoutError:
                event("tool_timeout", status="execution_unknown")
                return json.dumps({"status": "execution_unknown", "error": "工具执行超时，结果未知，请核实状态后再操作"}, ensure_ascii=False)
            except Exception as exc:
                failure("tool_failed", exc)
                return json.dumps({"status": "execution_unknown", "error": "工具调用异常，结果未知"}, ensure_ascii=False)

    def close(self):
        """清理 MCP 连接。"""
        if self._mcp_client:
            self._mcp_client.close()
            self._mcp_client = None
