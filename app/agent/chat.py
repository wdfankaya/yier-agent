import asyncio
import json
from typing import Optional

from openai import AsyncOpenAI

from app.agent.async_utils import run_sync
from app.agent.resilience import llm_create, llm_parse
from app.agent.storage import delete_session, load_session, save_session
from app.agent.summarizer import summarize
from app.config.settings import settings
from app.prompts.customer_service import SYSTEM_PROMPT
from app.schemas.response import CustomerServiceResponse, IntentType
from app.agent.tools.hitl import ConfirmationGate, notify_confirm_required, resolve_on_agent
from app.agent.tools.manager import ToolManager


class YierAgent:
    """电商客服 Agent —— AsyncOpenAI 全链路；chat() 仍是同步门面。"""

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
        # 按请求的 user_id 隔离长期记忆（默认仍取全局 settings.memory_user_id）
        memory_user_id = user_id or settings.memory_user_id
        self.user_id = memory_user_id  # 持久化按 (user_id, session_key) 定位会话行

        # memory/skill manager 先于 ToolManager 构造，注入给 ToolManager 按 agent 持有
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
        self.tool_manager = ToolManager(
            use_mcp=settings.mcp_enabled,
            mcp_server_url=settings.mcp_server_url,
            memory_manager=self.memory_manager if settings.memory_enabled else None,
            skill_manager=self.skill_manager if settings.skills_enabled else None,
            hitl=self.hitl,
        )

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None
        self._turns_since_ltm = 0
        self._ltm_tasks: set[asyncio.Task] = set()
        # chat() 走 asyncio.run，循环在返回后销毁；后台 create_task 会被取消，故改内联 await
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

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """同步门面：CLI / 测试 / 评估沙箱。服务端请 await achat()。"""
        self._ephemeral_loop = True
        try:
            return run_sync(self.achat(user_input))
        finally:
            self._ephemeral_loop = False

    async def achat(self, user_input: str) -> CustomerServiceResponse:
        """处理用户输入：ReAct 循环 → 结构化提取 → 返回结果（async）。"""
        self.hitl.consume_utterance(user_input)
        self.raw_messages.append({"role": "user", "content": user_input})

        final_text = await self._react_loop()

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

        # final 事件 = SSE 流的最后一帧（CLI 无回调，由 main.py 自己打印回复）
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
        """HTTP 确认入口：改闸；批准则当场执行敏感工具。"""
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
        """优雅关闭：收尾后台 LTM 任务 + 再巩固一次 + 关工具/HTTP 客户端。"""
        if self._ltm_tasks:
            await asyncio.gather(*list(self._ltm_tasks), return_exceptions=True)
        await self._aconsolidate_safe()
        self.tool_manager.close()
        await self.client.close()

    async def _maybe_consolidate(self) -> None:
        """每 N 轮巩固一次 LTM：服务端后台任务，CLI 内联 await（循环马上销毁）。"""
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
        except Exception as e:  # noqa: BLE001 —— 后台巩固失败不影响本轮对话
            print(f"⚠️  LTM 巩固失败: {type(e).__name__}: {e}")

    async def _react_loop(self) -> str:
        """ReAct 循环：调用 LLM → 执行工具 → 观察结果 → 重复，直到模型给出最终回答。"""
        for step in range(self.max_react_steps):
            messages = self._build_messages()

            response = await llm_create(
                self.client,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tools=self.tool_manager.tool_definitions,
            )
            choice = response.choices[0]
            assistant_msg = choice.message

            if assistant_msg.content:
                self._emit_thought(assistant_msg.content)

            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                self.raw_messages.append({"role": "assistant", "content": content})
                return content

            msg_dict = {"role": "assistant", "content": assistant_msg.content}
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant_msg.tool_calls
            ]
            self.raw_messages.append(msg_dict)

            for tc in assistant_msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                self._emit_tool_call(func_name, func_args)
                result_str = await self.tool_manager.aexecute_tool(func_name, func_args)
                self._emit_tool_result(func_name, result_str)

                self.raw_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        messages = self._build_messages()
        response = await llm_create(
            self.client,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        self.raw_messages.append({"role": "assistant", "content": content})
        return content

    async def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        """从最终文本中提取结构化元数据（意图、置信度等）。"""
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

    def _build_messages(self) -> list[dict]:
        system_content = SYSTEM_PROMPT
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(self.memory_manager.build_memory_prompt_sections())
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
                }
            )
        messages.extend(self.raw_messages)
        return messages

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

    def _emit(self, event_type: str, **payload) -> None:
        """事件统一出口：注入了 on_event 就推送结构化事件（SSE），否则静默。"""
        if self.on_event is not None:
            self.on_event({"type": event_type, **payload})

    def _emit_thought(self, text: str) -> None:
        """LLM 思考文本：SSE 推 thought 事件；CLI 保持原打印。"""
        if self.on_event is not None:
            self._emit("thought", content=text)
        else:
            print(f"\n💭 [思考] {text}")

    def _emit_tool_call(self, func_name: str, func_args: dict) -> None:
        """决定调用工具：SSE 推 tool_call；CLI 保持原打印。"""
        if self.on_event is not None:
            self._emit("tool_call", name=func_name, arguments=func_args)
        else:
            args_str = ", ".join(f"{k}={v!r}" for k, v in func_args.items())
            print(f"🔧 [调用工具] {func_name}({args_str})")

    def _emit_tool_result(self, func_name: str, result: str) -> None:
        """工具执行完：SSE 推完整结果的 tool_result；CLI 打印（截断到 300 字）。"""
        if self.on_event is not None:
            self._emit("tool_result", name=func_name, content=result)
        else:
            display = result if len(result) <= 300 else result[:300] + "..."
            print(f"📋 [工具结果] {display}")
        notify_confirm_required(self.on_event, self._emit, result)
