"""记忆管理器：统一管理短期记忆和长期记忆。

YierAgent 和 MultiAgentOrchestrator 通过此管理器与记忆系统交互。
"""

from __future__ import annotations

from typing import Any, Optional

from app.agent.async_utils import run_sync
from app.agent.memory.long_term import LongTermMemory
from app.agent.memory.short_term import ShortTermMemory


class MemoryManager:
    """记忆管理器：统一管理短期记忆和长期记忆。"""

    def __init__(
        self,
        client: Any,
        model: str,
        user_id: str = "default",
        memory_dir: str = "app/sessions/memory",
        memory_enabled: bool = True,
        max_ltm_facts: int = 50,
        source_session: str = "",
    ):
        self.client = client
        self.model = model
        self.memory_enabled = memory_enabled

        from uuid import uuid4
        self.source_session = source_session or uuid4().hex
        self.stm = ShortTermMemory(self.source_session)
        self.ltm = LongTermMemory(
            user_id=user_id,
            memory_dir=memory_dir,
            max_facts=max_ltm_facts,
            source_session=self.source_session,
        )

        if self.memory_enabled:
            self.ltm.load()

    async def aupdate_short_term(self, recent_messages: list[dict]) -> None:
        """每轮对话后更新短期记忆（async）。"""
        if not self.memory_enabled:
            return
        await self.stm.aupdate(self.client, self.model, recent_messages)
        await self.ltm.aextract_and_save(self.client, self.model, recent_messages, None)

    def update_short_term(self, recent_messages: list[dict]) -> None:
        """每轮对话后更新短期记忆（同步包装）。"""
        if not self.memory_enabled:
            return
        run_sync(self.aupdate_short_term(recent_messages))

    async def aprepare(self, query=""):
        import asyncio
        if self.memory_enabled:
            await asyncio.to_thread(self.ltm.refresh)
            await asyncio.to_thread(self.ltm.prepare_semantic, query)
            self.stm.reconcile(self.ltm.inspect(), self.ltm.reset_at)

    def build_memory_prompt_sections(self, query="") -> list[dict]:
        """生成所有记忆相关的 system prompt 消息列表。"""
        if not self.memory_enabled:
            return []

        sections = []
        ltm_section = self.ltm.build_prompt_section(query)
        if ltm_section:
            sections.append({"role": "system", "content": ltm_section, "_context_optional": True})
        stm_section = self.stm.build_prompt_section(query)
        if stm_section:
            sections.append({"role": "system", "content": stm_section, "_context_optional": True})
        return sections

    async def aconsolidate_to_long_term(
        self, messages: list[dict], summary: Optional[str],
    ) -> None:
        """将会话关键事实巩固到长期记忆（async）。"""
        if not self.memory_enabled:
            return
        await self.ltm.aextract_and_save(
            self.client, self.model, messages, summary,
        )

    def consolidate_to_long_term(
        self, messages: list[dict], summary: Optional[str],
    ) -> None:
        """会话结束时巩固长期记忆（同步包装：CLI / 测试）。"""
        if not self.memory_enabled:
            return
        run_sync(self.aconsolidate_to_long_term(messages, summary))

    def reset_short_term(self) -> None:
        """重置短期记忆（会话内重置时调用）。"""
        self.stm.reset()

    def reset_all(self) -> None:
        """重置所有记忆（短期+长期）。"""
        self.stm.reset()
        self.ltm.reset()

    def stm_to_dict(self) -> dict:
        return self.stm.to_dict()

    def restore_stm(self, data: dict) -> None:
        self.stm = ShortTermMemory.from_dict(data, self.source_session)
