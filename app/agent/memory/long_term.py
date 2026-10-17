"""长期记忆：跨会话持久化用户知识。

将用户在多次会话中表现出的偏好、身份、行为模式等提取并持久化为 JSON，
在新会话启动时加载并注入 prompt，让 Agent 具备"记住老客户"的能力。
"""

from __future__ import annotations

import json
import os
import uuid
from app.locks import KeyedLocks

_USER_LOCKS = KeyedLocks()
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.agent.memory.extraction import extract_long_term_facts


@dataclass
class MemoryFact:
    """一条长期记忆事实。"""
    content: str
    category: str  # identity / preference / behavior / issue / other
    created_at: str
    source_session: str = ""


class LongTermMemory:
    """跨会话长期记忆：持久化用户知识。"""

    def __init__(
        self,
        user_id: str = "default",
        memory_dir: str = "app/sessions/memory",
        max_facts: int = 50,
    ):
        self.user_id = user_id
        self.memory_dir = Path(memory_dir)
        self.max_facts = max_facts
        self._generation = 0
        self._known_facts = set()
        self._known_summaries = set()
        self.facts: list[MemoryFact] = []
        self.interaction_summaries: list[dict] = []

    @property
    def memory_path(self) -> Path:
        from app.server.identifiers import validate_id, safe_path
        return safe_path(self.memory_dir, f"{validate_id(self.user_id)}.json")

    def load(self) -> None:
        """从持久层加载用户的长期记忆（PG 或 JSON）。"""
        self.facts = []
        self.interaction_summaries = []
        if self._is_db():
            from app.server.repository import sync_load_facts
            facts, summaries = sync_load_facts(self.user_id)
            for item in facts:
                self.facts.append(MemoryFact(
                    content=item["content"],
                    category=item.get("category", "other"),
                    created_at=item.get("created_at", ""),
                    source_session="",
                ))
            self.interaction_summaries = [
                {"summary": s["content"], "timestamp": s["created_at"]}
                for s in summaries
            ]
            return

        if not self.memory_path.exists():
            return
        try:
            with self.memory_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        self._generation = data.get("generation", 0)
        for item in data.get("facts", []):
            self.facts.append(MemoryFact(
                content=item["content"],
                category=item.get("category", "other"),
                created_at=item.get("created_at", ""),
                source_session=item.get("source_session", ""),
            ))
        self.interaction_summaries = data.get("interaction_summaries", [])
        self._remember_snapshot()

    def save(self) -> None:
        """持久化到 PG 或 JSON 文件。"""
        if self._is_db():
            from app.server.repository import sync_upsert_facts
            sync_upsert_facts(
                self.user_id, "fact",
                [{"category": f.category, "content": f.content} for f in self.facts],
                keep_max=self.max_facts,
            )
            sync_upsert_facts(
                self.user_id, "session_summary",
                [{"category": "other", "content": s["summary"]}
                 for s in self.interaction_summaries],
            )
            return

        path = self.memory_path
        with _USER_LOCKS.hold(str(path)):
            disk = self._read_json()
            if disk.get("generation", 0) != self._generation:
                # A reset from another session wins over a stale snapshot.
                self.load()
                return
            additions = [f for f in self.facts if f.content.casefold() not in self._known_facts]
            summaries = [s for s in self.interaction_summaries if s["summary"] not in self._known_summaries]
            self.facts = [MemoryFact(**f) for f in disk.get("facts", [])]
            self.add_facts(additions)
            merged = {s["summary"]: s for s in disk.get("interaction_summaries", [])}
            merged.update({s["summary"]: s for s in summaries})
            self.interaction_summaries = list(merged.values())[-50:]
            self._write_json()
            self._remember_snapshot()

    def _read_json(self):
        if not self.memory_path.exists():
            return {}
        # Do not silently overwrite a corrupt existing file.
        return json.loads(self.memory_path.read_text(encoding="utf-8"))

    def _remember_snapshot(self):
        self._known_facts = {f.content.casefold() for f in self.facts}
        self._known_summaries = {s["summary"] for s in self.interaction_summaries}

    def _write_json(self):
        path = self.memory_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "generation": self._generation, "user_id": self.user_id,
                   "updated_at": datetime.now().isoformat(timespec="seconds"),
                   "facts": [asdict(f) for f in self.facts],
                   "interaction_summaries": self.interaction_summaries}
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def add_facts(self, new_facts: list[MemoryFact]) -> None:
        """添加新事实，自动去重并裁剪到 max_facts。"""
        existing_contents = {f.content.casefold() for f in self.facts}
        for fact in new_facts:
            if fact.content.casefold() not in existing_contents:
                self.facts.append(fact)
                existing_contents.add(fact.content.casefold())

        if len(self.facts) > self.max_facts:
            self.facts = self.facts[-self.max_facts:]

    def add_interaction_summary(self, summary: str) -> None:
        self.interaction_summaries.append({
            "summary": summary,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    async def aextract_and_save(
        self,
        client: Any,
        model: str,
        messages: list[dict],
        summary: Optional[str],
    ) -> None:
        """从会话消息中提取长期记忆事实并保存（LLM await，落盘下沉线程池）。"""
        if not messages and not summary:
            return

        new_facts, interaction_summary = await extract_long_term_facts(
            client, model, messages, summary, self.facts,
        )

        if new_facts:
            self.add_facts(new_facts)
        if interaction_summary:
            self.add_interaction_summary(interaction_summary)

        # save() 内部可能 asyncio.run(PG)，不能在已经在跑的 loop 里直接调
        import asyncio
        await asyncio.to_thread(self.save)

    def extract_and_save(
        self,
        client: Any,
        model: str,
        messages: list[dict],
        summary: Optional[str],
    ) -> None:
        from app.agent.async_utils import run_sync
        run_sync(self.aextract_and_save(client, model, messages, summary))

    def build_prompt_section(self) -> str | None:
        """生成注入 system prompt 的长期记忆片段。"""
        if not self.facts and not self.interaction_summaries:
            return None

        parts = []
        if self.facts:
            facts_text = "\n".join(
                f"- [{f.category}] {f.content}" for f in self.facts
            )
            parts.append(f"该用户的历史记忆（来自过往会话）：\n{facts_text}")

        if self.interaction_summaries:
            recent = self.interaction_summaries[-3:]
            summaries_text = "\n".join(f"- {s['summary']}" for s in recent)
            parts.append(f"最近的交互记录：\n{summaries_text}")

        return "\n\n".join(parts)

    def reset(self) -> None:
        """清空该用户的长期记忆（DB 删该 user 全部 memory_facts；JSON 删文件）。"""
        if self._is_db():
            from app.server.repository import sync_delete_user_memory
            sync_delete_user_memory(self.user_id)
            self.facts = []
            self.interaction_summaries = []
            return
        with _USER_LOCKS.hold(str(self.memory_path)):
            self._generation = self._read_json().get("generation", 0) + 1
            self.facts = []
            self.interaction_summaries = []
            self._write_json()
            self._remember_snapshot()

    @staticmethod
    def _is_db() -> bool:
        """PG 模式开关（与 storage.py 同源，读 settings.db_enabled）。"""
        from app.config.settings import settings
        return settings.db_enabled
