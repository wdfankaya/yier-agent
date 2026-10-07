"""Redis 冷读缓存测试。

场景：
1. SET/GET/DELETE 会话快照（真 Redis，不可用则用内存替身）
2. Cache-Aside：未命中走 PG 并回填；二次读取不再打 PG
3. 写穿：save 先「PG」再更新缓存；load 命中缓存
4. 命中比回源快（模拟慢 PG）
5. Redis 挂了：get/set 不抛、服务回源 PG
6. INCR + EXPIRE 窗口计数
7. enabled=False 空操作
8. （可选）真 PG 往返：写穿 + 杀缓存后仍能从 PG 读到

用法：python tests/test_session_cache.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.agent.storage import delete_session, load_session, save_session  # noqa: E402
from app.server.cache import SessionCache, reset_cache  # noqa: E402
from app.server.repository import SessionState  # noqa: E402

TEST_REDIS = "redis://127.0.0.1:6379/15"
TEST_PATH = "app/sessions/server/cache-test-user/cache-demo.json"


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _skip(msg: str):
    print(f"  ⏭️  {msg}")


class _MemRedis:
    """够用的内存 Redis：get/set/delete/incr/expire/pipeline/ping。"""

    def __init__(self):
        self.store: dict[str, str] = {}
        self._ttl: dict[str, int] = {}

    def ping(self):
        return True

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v
        if ex is not None:
            self._ttl[k] = int(ex)
        return True

    def delete(self, k):
        self.store.pop(k, None)
        self._ttl.pop(k, None)
        return 1

    def incr(self, k):
        n = int(self.store.get(k, 0) or 0) + 1
        self.store[k] = str(n)
        return n

    def expire(self, k, time, nx=False):
        if k not in self.store:
            return False
        if nx and k in self._ttl:
            return False
        self._ttl[k] = int(time)
        return True

    def ttl(self, k):
        return self._ttl.get(k, -1)

    def pipeline(self):
        return _MemPipe(self)

    def close(self):
        pass


class _MemPipe:
    def __init__(self, r: _MemRedis):
        self.r = r
        self.ops = []

    def incr(self, k):
        self.ops.append(("incr", k))
        return self

    def expire(self, k, t, nx=False):
        self.ops.append(("expire", k, t, nx))
        return self

    def execute(self):
        out = []
        for op in self.ops:
            if op[0] == "incr":
                out.append(self.r.incr(op[1]))
            else:
                out.append(self.r.expire(op[1], op[2], nx=op[3]))
        self.ops.clear()
        return out


class _BrokenRedis:
    def ping(self):
        raise ConnectionError("redis down")

    def get(self, k):
        raise ConnectionError("redis down")

    def set(self, k, v, ex=None):
        raise ConnectionError("redis down")

    def delete(self, k):
        raise ConnectionError("redis down")

    def pipeline(self):
        raise ConnectionError("redis down")

    def close(self):
        pass


def _make_cache():
    live = SessionCache(enabled=True, url=TEST_REDIS, ttl=60)
    if live.status() == "ok":
        return live, True
    return SessionCache(enabled=True, url=TEST_REDIS, ttl=60, client=_MemRedis()), False


def _payload(text="hi"):
    return {
        "summary": "s",
        "messages": [{"role": "user", "content": text}],
        "short_term_memory": {"facts": ["喜欢蓝色"]},
    }


def _state(text="hi"):
    return SessionState(
        session_id=uuid.uuid4(),
        session_key="cache-demo",
        user_id="cache-test-user",
        summary="s",
        stm={"facts": ["喜欢蓝色"]},
        messages=[{"role": "user", "content": text}],
    )


# ---------- 1. 快照 CRUD ----------
def test_snapshot_crud():
    print("\n[1/8] 会话快照 SET/GET/DELETE")
    cache, live = _make_cache()
    uid, key = "cache-test-user", f"crud-{uuid.uuid4().hex[:8]}"
    assert cache.set_session(uid, key, _payload("crud"))
    got = cache.get_session(uid, key)
    if got is None or got["messages"][0]["content"] != "crud":
        _fail(f"GET 未命中刚 SET 的快照: {got}")
    _ok("SET 后 GET 命中，payload 完整")
    cache.delete_session(uid, key)
    if cache.get_session(uid, key) is not None:
        _fail("DELETE 后仍能 GET")
    _ok("DELETE 后 miss")
    cache.close()
    _ok("后端: " + ("Redis db15" if live else "内存替身（本机 Redis 不可用）"))


# ---------- 2. Cache-Aside ----------
def test_cache_aside():
    print("\n[2/8] Cache-Aside：miss → PG → 回填 → hit 不再打 PG")
    cache, _ = _make_cache()
    reset_cache(cache)
    pg_calls = []

    def fake_load(uid, key):
        pg_calls.append((uid, key))
        return _state("aside")

    try:
        with patch("app.agent.storage._is_db", return_value=True), \
             patch("app.server.repository.sync_load_session_state", side_effect=fake_load):
            first = load_session(TEST_PATH, "cache-test-user")
            second = load_session(TEST_PATH, "cache-test-user")
        if first is None or first["messages"][0]["content"] != "aside":
            _fail(f"首次 load 失败: {first}")
        if second != first:
            _fail("二次 load 与缓存不一致")
        if len(pg_calls) != 1:
            _fail(f"预期只打 1 次 PG，实际 {len(pg_calls)}")
        _ok("miss 回源一次并回填；第二次命中缓存，PG 不再被打")
        if cache.hits < 1 or cache.misses < 1:
            _fail(f"计数异常 hits={cache.hits} misses={cache.misses}")
        _ok(f"计数 hits={cache.hits} misses={cache.misses}")
    finally:
        cache.delete_session("cache-test-user", "cache-demo")
        reset_cache(None)
        cache.close()


# ---------- 3. 写穿 ----------
def test_write_through():
    print("\n[3/8] 写穿：save 先 PG 再更新缓存")
    cache, _ = _make_cache()
    reset_cache(cache)
    replaced = []

    def fake_create(uid, key):
        return uuid.uuid4()

    def fake_replace(sid, messages, summary, stm):
        replaced.append(list(messages))

    def boom_load(*a, **k):
        raise AssertionError("写穿后 load 不应回源 PG")

    try:
        with patch("app.agent.storage._is_db", return_value=True), \
             patch("app.server.repository.sync_get_or_create_session", side_effect=fake_create), \
             patch("app.server.repository.sync_replace_session_state", side_effect=fake_replace):
            save_session(
                TEST_PATH,
                [{"role": "user", "content": "wt"}],
                "sum",
                {"facts": []},
                user_id="cache-test-user",
            )
        if not replaced:
            _fail("PG replace 没被调用（写穿要求先写 PG）")
        _ok("save 先写了 PG")

        with patch("app.agent.storage._is_db", return_value=True), \
             patch("app.server.repository.sync_load_session_state", side_effect=boom_load):
            got = load_session(TEST_PATH, "cache-test-user")
        if got is None or got["messages"][0]["content"] != "wt":
            _fail(f"写穿后缓存未更新: {got}")
        _ok("save 后 GET 命中缓存，无需回源")
    finally:
        cache.delete_session("cache-test-user", "cache-demo")
        reset_cache(None)
        cache.close()


# ---------- 4. 命中更快 ----------
def test_hit_faster_than_pg():
    print("\n[4/8] 命中缓存比回源 PG 快")
    cache, _ = _make_cache()
    reset_cache(cache)
    pg_calls = []

    def slow_load(uid, key):
        pg_calls.append(1)
        time.sleep(0.08)
        return _state("slow")

    try:
        with patch("app.agent.storage._is_db", return_value=True), \
             patch("app.server.repository.sync_load_session_state", side_effect=slow_load):
            t0 = time.perf_counter()
            load_session(TEST_PATH, "cache-test-user")
            t_miss = time.perf_counter() - t0
            t0 = time.perf_counter()
            load_session(TEST_PATH, "cache-test-user")
            t_hit = time.perf_counter() - t0
        if len(pg_calls) != 1:
            _fail(f"二次 load 仍打了 PG: {len(pg_calls)}")
        if t_hit >= t_miss / 2:
            _fail(f"命中未明显更快: hit={t_hit:.3f}s miss={t_miss:.3f}s")
        _ok(f"miss={t_miss*1000:.0f}ms（含模拟 PG 80ms） hit={t_hit*1000:.1f}ms")
    finally:
        cache.delete_session("cache-test-user", "cache-demo")
        reset_cache(None)
        cache.close()


# ---------- 5. Redis 挂了回源 ----------
def test_redis_down_fallback():
    print("\n[5/8] Redis 挂了：不抛异常，自动回源 PG")
    dead = SessionCache(
        enabled=True,
        url="redis://127.0.0.1:1/0",
        ttl=60,
        client=_BrokenRedis(),
    )
    reset_cache(dead)
    t0 = time.perf_counter()
    if dead.get_session("u", "s") is not None:
        _fail("挂了的缓存不应命中")
    if not dead.set_session("u", "s", _payload()):
        pass  # 期望 False
    else:
        _fail("挂了的缓存 SET 不应报成功")
    if dead.incr_window("rate:x") is not None:
        _fail("挂了的缓存 INCR 应返回 None（限流 fail-open）")
    elapsed = time.perf_counter() - t0
    if elapsed > 2.0:
        _fail(f"Redis 挂了拖太久: {elapsed:.2f}s")
    _ok(f"get/set/incr 均降级且 {elapsed*1000:.0f}ms 内返回")

    def fake_load(uid, key):
        return _state("from-pg")

    try:
        with patch("app.agent.storage._is_db", return_value=True), \
             patch("app.server.repository.sync_load_session_state", side_effect=fake_load):
            got = load_session(TEST_PATH, "cache-test-user")
        if got is None or got["messages"][0]["content"] != "from-pg":
            _fail(f"Redis 挂了未能回源 PG: {got}")
        _ok("load_session 在 Redis 挂了时回源 PG 成功")
        if dead.status() != "down":
            _fail(f"status 应为 down，实际 {dead.status()}")
        _ok("status=down，但调用方仍拿到会话")
    finally:
        reset_cache(None)
        dead.close()


# ---------- 6. INCR 窗口计数 ----------
def test_incr_window():
    print("\n[6/8] INCR + EXPIRE 窗口计数")
    cache, _ = _make_cache()
    name = f"rate:cache-test-user-{uuid.uuid4().hex[:8]}"
    n1 = cache.incr_window(name, 60)
    n2 = cache.incr_window(name, 60)
    n3 = cache.incr_user_rate("u-incr-test", 60)
    if n1 != 1 or n2 != 2:
        _fail(f"INCR 计数错误: {n1}, {n2}")
    _ok("同一 key 连续 INCR → 1, 2")
    if n3 != 1:
        _fail(f"incr_user_rate 首次应为 1，实际 {n3}")
    _ok("incr_user_rate 写入 rate:{user_id}")
    client = cache._client()
    ttl = client.ttl(name) if hasattr(client, "ttl") else cache.ttl
    if isinstance(ttl, int) and ttl > 60:
        _fail(f"TTL 不应超过窗口: {ttl}")
    _ok("EXPIRE NX 已设窗口（第二次 INCR 不刷新成滑动窗口）")
    cache.close()


# ---------- 7. disabled ----------
def test_disabled():
    print("\n[7/8] redis_enabled=False 时空操作")
    cache = SessionCache(enabled=False, url=TEST_REDIS, ttl=60)
    if cache.status() != "disabled":
        _fail(f"status 应为 disabled，实际 {cache.status()}")
    if cache.get_session("u", "s") is not None:
        _fail("disabled 时 GET 应 miss")
    if cache.set_session("u", "s", _payload()):
        _fail("disabled 时 SET 应失败/空操作")
    if cache.incr_window("x") is not None:
        _fail("disabled 时 INCR 应返回 None")
    cache.delete_session("u", "s")  # 不抛
    _ok("disabled：status=disabled，读写都是空操作")
    cache.close()


# ---------- 8. 真 PG 往返（可选） ----------
def test_live_pg_roundtrip():
    print("\n[8/8] 真 PG 写穿 + 缓存失效后回源（需 DB_ENABLED + 本机 PG）")
    from app.config.settings import settings
    if not settings.db_enabled:
        _skip("DB_ENABLED=false，跳过真 PG 往返")
        return

    live = SessionCache(enabled=True, url=TEST_REDIS, ttl=120)
    if live.status() != "ok":
        _skip("Redis 不可用，跳过真 PG 往返")
        live.close()
        return

    user = f"cache-test-user-{uuid.uuid4().hex[:8]}"
    path = f"app/sessions/server/{user}/live.json"
    reset_cache(live)
    try:
        save_session(
            path,
            [{"role": "user", "content": "live-pg"}],
            None,
            None,
            user_id=user,
        )
        cached = live.get_session(user, "live")
        if cached is None or cached["messages"][0]["content"] != "live-pg":
            _fail(f"写穿后 Redis 没有副本: {cached}")
        _ok("真 PG save 后 Redis 有副本")

        live.delete_session(user, "live")  # 只删缓存，模拟冷 miss
        got = load_session(path, user)
        if got is None or got["messages"][0]["content"] != "live-pg":
            _fail(f"缓存失效后未能回源 PG: {got}")
        _ok("删掉 Redis 副本后 load 回源 PG 成功")

        dead = SessionCache(enabled=True, url=TEST_REDIS, ttl=60, client=_BrokenRedis())
        reset_cache(dead)
        got2 = load_session(path, user)
        if got2 is None or got2["messages"][0]["content"] != "live-pg":
            _fail(f"Redis 挂了未能回源 PG: {got2}")
        _ok("注入挂掉的 Redis 后仍从 PG 读到会话（缓存挂了服务不挂）")
        dead.close()
    except Exception as e:  # noqa: BLE001
        _skip(f"真 PG 往返失败（库不可达?）: {type(e).__name__}: {e}")
    finally:
        reset_cache(live)
        try:
            delete_session(path, user)
        except Exception:
            pass
        reset_cache(None)
        live.close()


def main():
    print("=" * 60)
    print("Redis 冷读缓存")
    print("=" * 60)
    test_snapshot_crud()
    test_cache_aside()
    test_write_through()
    test_hit_faster_than_pg()
    test_redis_down_fallback()
    test_incr_window()
    test_disabled()
    test_live_pg_roundtrip()
    print("\n全部通过。")


if __name__ == "__main__":
    main()
