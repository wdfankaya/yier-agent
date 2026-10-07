"""会话持久化统一入口（JSON → PG 的过渡层）。

保留三个函数的「接口形状」（save_session / load_session / delete_session），
内部按 settings.db_enabled 分流：
- db_enabled=False（CLI / 离线 eval）：沿用原有 JSON 原子写盘；
- db_enabled=True（服务端 / PG）：转调 app.server.repository 的 sync 层，
  session_key 由 path 的文件名推导（server 路径 app/sessions/server/{user}/{key}.json → key）。

db_enabled=True 时在 PG 之上叠一层 Redis 冷读缓存（Cache-Aside + 写穿）。
Redis 不是真相源，挂了自动回源 PG。JSON 回退路径不走 Redis。

这样 chat.py / orchestrator 不用感知底层，改动面收敛在这一层。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

SESSION_VERSION = 1


def _session_payload(messages, summary, short_term_memory) -> dict:
    return {
        "summary": summary,
        "messages": messages,
        "short_term_memory": short_term_memory,
    }


def _cache():
    from app.server.cache import get_cache
    return get_cache()


def _is_db() -> bool:
    from app.config.settings import settings
    return settings.db_enabled


def _db_key(path: str) -> str:
    return Path(path).stem


def save_session(
    path: str,
    messages: list[dict],
    summary: Optional[str],
    short_term_memory: Optional[dict] = None,
    user_id: Optional[str] = None,
) -> None:
    """把对话状态写入持久层（PG 或 JSON 文件）。

    messages 只包含原始 user/assistant 条目（不含 system / summary）。
    short_term_memory 为短期记忆的序列化数据。
    """
    if _is_db():
        from app.config.settings import settings
        from app.server.repository import (
            sync_get_or_create_session,
            sync_replace_session_state,
        )
        uid = user_id or settings.memory_user_id
        key = _db_key(path)
        sid = sync_get_or_create_session(uid, key)
        sync_replace_session_state(sid, messages, summary, short_term_memory or None)
        # 写穿：PG 是真相源，成功后再更新缓存副本；Redis 失败忽略。
        _cache().set_session(
            uid, key, _session_payload(messages, summary, short_term_memory),
        )
        return

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": SESSION_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "messages": messages,
        "short_term_memory": short_term_memory,
    }

    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)


def load_session(
    path: str,
    user_id: Optional[str] = None,
) -> Optional[dict]:
    """读取会话状态。不存在或损坏都返回 None（降级为新会话）。"""
    if _is_db():
        from app.config.settings import settings
        from app.server.repository import sync_load_session_state
        uid = user_id or settings.memory_user_id
        key = _db_key(path)
        cached = _cache().get_session(uid, key)
        if cached is not None:
            return cached
        st = sync_load_session_state(uid, key)
        if st is None:
            return None
        payload = _session_payload(st.messages, st.summary, st.stm)
        _cache().set_session(uid, key, payload)  # Cache-Aside 回填
        return payload

    file_path = Path(path)
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  会话文件损坏，已忽略（{e}）")
        return None

    if not isinstance(data, dict) or "messages" not in data:
        print("⚠️  会话文件格式不识别，已忽略")
        return None

    return {
        "summary": data.get("summary"),
        "messages": data.get("messages", []),
        "short_term_memory": data.get("short_term_memory"),
    }


def delete_session(path: str, user_id: Optional[str] = None) -> None:
    """删除会话持久状态（PG 删行级联清消息；JSON 删文件），不存在时静默。"""
    if _is_db():
        from app.config.settings import settings
        from app.server.repository import sync_delete_session_state
        uid = user_id or settings.memory_user_id
        key = _db_key(path)
        sync_delete_session_state(uid, key)
        _cache().delete_session(uid, key)
        return

    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()
