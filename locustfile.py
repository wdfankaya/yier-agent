"""Locust 压测场景。

任务配比：对话 SSE（读到流结束）: 查历史 = 3:1。
每个虚拟用户有自己的 session_id / user_id，避免热会话互相覆盖 on_event。

用法（先起服务）：
  .venv/Scripts/python -m app.server.main
  locust -f locustfile.py --host http://127.0.0.1:8000

摸底基线（LLM 打桩、不含真实模型）：
  python -m app.scripts.run_load_baseline
"""

from __future__ import annotations

import json
import uuid

from locust import HttpUser, between, task


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        line = next((ln for ln in block.split("\n") if ln.startswith("data: ")), None)
        if not line:
            continue
        try:
            events.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            continue
    return events


class YierUser(HttpUser):
    wait_time = between(0.2, 0.6)

    def on_start(self):
        suffix = uuid.uuid4().hex[:8]
        self.session_id = f"locust-{suffix}"
        self.user_id = f"lu-{suffix}"

    @task(3)
    def chat_sse(self):
        with self.client.post(
            "/api/chat",
            json={
                "session_id": self.session_id,
                "user_id": self.user_id,
                "message": "你好，在吗",
            },
            name="POST /api/chat (SSE)",
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code == 429:
                resp.failure("rate limited")
                return
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            events = _parse_sse(resp.text)
            types = [e.get("type") for e in events]
            if "final" not in types and "error" not in types:
                resp.failure(f"SSE 未结束: {types[:8]}")
                return
            sess = next((e for e in events if e.get("type") == "session"), None)
            if sess and sess.get("session_id") != self.session_id:
                resp.failure("session_id mismatch")
                return
            resp.success()

    @task(1)
    def get_history(self):
        with self.client.get(
            f"/api/sessions/{self.session_id}",
            name="GET /api/sessions/:id",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")
