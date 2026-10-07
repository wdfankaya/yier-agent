"""Multi-Agent 编排器：协调 Router 和子 Agent 完成用户请求。

流程：Router 分类意图 → 选择子 Agent → ReAct 执行 → 结构化提取 → 持久化。
"""

import asyncio
from typing import Optional

from openai import AsyncOpenAI

from app.agent.async_utils import run_sync
from app.agent.resilience import llm_create, llm_parse
from app.agent.storage import delete_session, load_session, save_session
from app.agent.summarizer import summarize
from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS, SubAgent
from app.multi_agent.router import Router
from app.schemas.response import CustomerServiceResponse, IntentType
from app.agent.tools.hitl import ConfirmationGate, resolve_on_agent
from app.agent.tools.manager import ToolManager


class MultiAgentOrchestrator:
    """多 Agent 编排器，对外接口与 YierAgent 一致。"""

    def __init__(
        self,
        session_path: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_react_steps = settings.max_react_steps
        # 事件出口。None = CLI（保留原 print）；SSE 时由服务端注入回调
        self.on_event = None
        # 按请求的 user_id 隔离长期记忆
        memory_user_id = user_id or settings.memory_user_id
        self.user_id = memory_user_id  # 持久化按 (user_id, session_key) 定位会话行

        self.router = Router(self.client, self.model)

        # memory/skill manager 先构造，再注入到各子 Agent 的 ToolManager（按 agent 持有）
        from app.agent.memory import MemoryManager
        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=memory_user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=settings.memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )

        from app.agent.skills import SkillManager
        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )

        self.hitl = ConfirmationGate()
        self.agents: dict[str, SubAgent] = {}
        for key, cfg in AGENT_CONFIGS.items():
            tm = ToolManager(
                use_mcp=settings.mcp_enabled,
                mcp_server_url=settings.mcp_server_url,
                allowed_tools=cfg["tools"],
                memory_manager=self.memory_manager if settings.memory_enabled else None,
                skill_manager=self.skill_manager if settings.skills_enabled else None,
                hitl=self.hitl,
            )
            self.agents[key] = SubAgent(
                name=cfg["name"],
                system_prompt=cfg["prompt"],
                tool_manager=tm,
                client=self.client,
                model=self.model,
                temperature=self.temperature,
            )

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        self._turns_since_ltm = 0
        self._ltm_tasks: set[asyncio.Task] = set()
        self._ephemeral_loop = False

        loaded = load_session(self.session_path, self.user_id)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    def _emit(self, event_type: str, **payload) -> None:
        """事件统一出口：注入了 on_event 就推送结构化事件（SSE），否则静默。"""
        if self.on_event is not None:
            self.on_event({"type": event_type, **payload})

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """同步门面：CLI / 测试 / 评估沙箱。服务端请 await achat()。"""
        self._ephemeral_loop = True
        try:
            return run_sync(self.achat(user_input))
        finally:
            self._ephemeral_loop = False

    async def achat(self, user_input: str) -> CustomerServiceResponse:
        """路由 → 子 Agent 执行 → 结构化提取 → 返回结果。"""
        self.hitl.consume_utterance(user_input)
        self.raw_messages.append({"role": "user", "content": user_input})

        agent_key = await self.router.aroute(user_input, self.raw_messages)
        agent = self.agents[agent_key]
        if self.on_event is not None:
            self._emit("route", agent=agent.name)
        else:
            print(f"\n🔀 [路由] → {agent.name}")

        messages = self._build_messages(agent)
        # 把编排器的事件出口透传给被路由的子 Agent，其 thought/tool 事件才能上流
        agent.on_event = self.on_event
        try:
            final_text, new_messages = await agent.ahandle(
                messages, max_steps=self.max_react_steps,
            )
        finally:
            agent.on_event = None
        self.raw_messages.extend(new_messages)

        result = await self._extract_structured_response(final_text)

        await self.memory_manager.aupdate_short_term(self.raw_messages[-6:])

        self.raw_messages.append(
            {"role": "assistant", "content": result.model_dump_json(ensure_ascii=False)}
        )

        if len(self.raw_messages) > self.history_threshold:
            await self._compress_history()

        await asyncio.to_thread(
            save_session,
            self.session_path, self.raw_messages, self.summary,
            self.memory_manager.stm_to_dict(),
            self.user_id,
        )

        await self._maybe_consolidate()

        # final 事件 = SSE 流的最后一帧
        self._emit(
            "final",
            reply=result.reply,
            intent=result.intent.value,
            confidence=result.confidence,
            requires_human=result.requires_human,
        )
        return result

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.hitl.reset()
        self.memory_manager.reset_short_term()
        delete_session(self.session_path, self.user_id)

    def resolve_hitl(self, token: str, approved: bool) -> dict:
        """HTTP 确认入口：会话级闸门，批准后由持有该工具的子 Agent 执行。"""
        return resolve_on_agent(self, token, approved)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
            user_id=self.user_id,
        )

    def close(self):
        run_sync(self.aclose())

    async def aclose(self):
        if self._ltm_tasks:
            await asyncio.gather(*list(self._ltm_tasks), return_exceptions=True)
        await self._aconsolidate_safe()
        for agent in self.agents.values():
            agent.tool_manager.close()
        await self.client.close()

    async def _maybe_consolidate(self) -> None:
        n = settings.ltm_consolidate_every
        if n <= 0 or not settings.memory_enabled:
            return
        self._turns_since_ltm += 1
        if self._turns_since_ltm < n:
            return
        self._turns_since_ltm = 0
        if self._ephemeral_loop:
            await self._aconsolidate_safe()
            return
        task = asyncio.get_running_loop().create_task(self._aconsolidate_safe())
        self._ltm_tasks.add(task)
        task.add_done_callback(self._ltm_tasks.discard)

    async def _aconsolidate_safe(self) -> None:
        try:
            await self.memory_manager.aconsolidate_to_long_term(
                list(self.raw_messages), self.summary,
            )
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  LTM 巩固失败: {type(e).__name__}: {e}")

    def _build_messages(self, agent: SubAgent) -> list[dict]:
        """用子 Agent 的 system prompt 构建消息列表。"""
        system_content = agent.system_prompt
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(self.memory_manager.build_memory_prompt_sections())
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
            })
        messages.extend(self.raw_messages)
        return messages

    async def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        try:
            response = await llm_parse(
                self.client,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "基于以下客服回复内容，提取结构化信息。"
                            "reply 字段直接使用原文，不要修改或缩减。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                response_format=CustomerServiceResponse,
            )
            return response.choices[0].message.parsed
        except Exception:
            return await self._extract_structured_fallback(text)

    async def _extract_structured_fallback(self, text: str) -> CustomerServiceResponse:
        """当 response_format 不被 API 支持时，用 prompt 引导 JSON 输出。"""
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        response = await llm_create(
            self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "基于以下客服回复内容，提取结构化信息并输出 JSON。\n"
                        "reply 字段直接使用原文，不要修改或缩减。\n\n"
                        "必须严格按照以下 JSON 格式输出（不要加 markdown 代码块）：\n"
                        "{\n"
                        f'  "intent": <从以下选择: {intent_values}>,\n'
                        '  "confidence": <0.0到1.0的浮点数>,\n'
                        '  "reply": <原文回复内容>,\n'
                        '  "requires_human": <true或false>,\n'
                        '  "follow_up_question": <追问问题或null>\n'
                        "}"
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return CustomerServiceResponse.model_validate_json(raw)

    async def _compress_history(self) -> None:
        keep = self.history_keep_recent
        split = len(self.raw_messages) - keep
        while split > 0 and self.raw_messages[split].get("role") in ("tool",):
            split -= 1
        if split <= 0:
            return
        old_messages = self.raw_messages[:split]
        recent = self.raw_messages[split:]

        new_summary = await summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.summary = new_summary
        self.raw_messages = recent
        print(
            f"\n💾 [已压缩 {len(old_messages)} 条老消息 → summary "
            f"({len(new_summary)} 字)]\n"
        )
