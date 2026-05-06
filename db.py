# -*- coding: utf-8 -*-
"""
SQLite 持久化模块
用于存储审计日志、违禁词记录、违规事件
A100 优化：连接池、WAL 模式、批量写入
"""
import aiosqlite
import asyncio
from pathlib import Path
from typing import Optional, Callable, TypeVar

T = TypeVar("T")

DB_PATH = Path(__file__).parent.parent / "data" / "audit.db"


class _DBPool:
    """轻量级 aiosqlite 连接池。"""
    def __init__(self, size: int = 4):
        self._size = size
        self._pool: list[aiosqlite.Connection] = []
        self._lock = asyncio.Lock()
        self._ready = False

    async def _create_conn(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(str(DB_PATH))
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=-64000")
        await conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    async def init(self):
        """创建表结构并预热连接池。需在 lifespan 中调用一次。"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = await self._create_conn()
        try:
            await conn.executescript("""
                CREATE TABLE IF NOT EXISTS rooms (
                    room_id TEXT PRIMARY KEY, room_name TEXT, platform TEXT, streamer TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, room_id TEXT, client_ip TEXT,
                    started_at DATETIME DEFAULT CURRENT_TIMESTAMP, ended_at DATETIME,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (room_id) REFERENCES rooms(room_id)
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT, room_id TEXT, asr_text TEXT,
                    status TEXT, risk_level TEXT, source TEXT,
                    matched_word TEXT, segment_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS forbidden_words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT NOT NULL,
                    level TEXT DEFAULT 'high', category TEXT, note TEXT,
                    enabled INTEGER DEFAULT 1, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_forbidden_word ON forbidden_words(word);
                CREATE INDEX IF NOT EXISTS idx_forbidden_enabled ON forbidden_words(enabled);
                CREATE INDEX IF NOT EXISTS idx_audit_room ON audit_log(room_id);
                CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status);
                CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(timestamp);
            """)
            async with conn.execute("PRAGMA table_info(audit_log)") as cur:
                cols = [row[1] for row in await cur.fetchall()]
            if "segment_id" not in cols:
                await conn.execute("ALTER TABLE audit_log ADD COLUMN segment_id TEXT")
            await conn.commit()
        finally:
            await conn.close()
        for _ in range(self._size):
            self._pool.append(await self._create_conn())
        self._ready = True
        print(f"--- [数据库] 初始化完成: {DB_PATH} ---")

    async def acquire(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._pool:
                return self._pool.pop()
        return await self._create_conn()

    async def release(self, conn: aiosqlite.Connection):
        async with self._lock:
            if len(self._pool) < self._size:
                try:
                    await conn.execute("SELECT 1")
                except Exception:
                    await conn.close()
                    return
                self._pool.append(conn)
            else:
                await conn.close()

    async def close_all(self):
        async with self._lock:
            for c in self._pool:
                await c.close()
            self._pool.clear()


_pool: Optional[_DBPool] = None


async def get_pool() -> _DBPool:
    global _pool
    if _pool is None:
        _pool = _DBPool(size=4)
    return _pool


def _run(fn: Callable[[aiosqlite.Connection], T]) -> T:
    """从连接池获取一个连接执行 fn（含 COMMIT），完成后归还。fn 接收 Connection 对象。"""
    loop = asyncio.get_running_loop()
    # 同步包装：在 executor 中运行，避免阻塞事件循环（单连接 SQLite 读操作快，但安全起见）
    async def _go():
        pool = await get_pool()
        conn = await pool.acquire()
        try:
            return await fn(conn)
        finally:
            await pool.release(conn)
    return loop.create_task(_go())  # type: ignore


# ================= helpers =================

def _sess_fn(session_id: str, room_id: str, client_ip: str):
    """建 lambda 的闭包：插入会话。"""
    def _do(c):
        return c.execute(
            "INSERT OR IGNORE INTO sessions (session_id,room_id,client_ip) VALUES (?,?,?)",
            (session_id, room_id, client_ip))
    return _do


def _log_fn(sid, rid, text, status, rl, src, mw="", seg=""):
    def _do(c):
        return c.execute(
            """INSERT INTO audit_log
               (session_id,room_id,asr_text,status,risk_level,source,matched_word,segment_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            (sid, rid, text, status, rl, src, mw, seg or ""))
    return _do


# ================= public API =================

async def init_db():
    pool = await get_pool()
    if not pool._ready:
        await pool.init()


async def save_session(session_id: str, room_id: str, client_ip: str):
    await _run(_sess_fn(session_id, room_id, client_ip))
    # 需要 commit —— 用同一个连接池的 acquire+commit+release 模式
    await _exec_commit(lambda c: c.commit())


async def end_session(session_id: str):
    await _exec(lambda c: c.execute(
        "UPDATE sessions SET ended_at=CURRENT_TIMESTAMP,status='closed' WHERE session_id=?",
        (session_id,)))
    await _exec_commit(lambda c: c.commit())


async def log_audit(
    session_id: str, room_id: str, text: str, status: str,
    risk_level: str, source: str, matched_word: str = "", segment_id: str = "",
):
    await _exec(lambda c: c.execute(
        """INSERT INTO audit_log
           (session_id,room_id,asr_text,status,risk_level,source,matched_word,segment_id)
           VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, room_id, text, status, risk_level, source, matched_word, segment_id or "")))
    await _exec_commit(lambda c: c.commit())


async def upsert_forbidden_words(words: list[dict]):
    """批量插入或更新违禁词。"""
    def _do(c):
        c.execute("UPDATE forbidden_words SET enabled=0")
        upd = [(w.get("level","high"), w.get("category",""), w.get("note",""), w["word"]) for w in words]
        c.executemany(
            "UPDATE forbidden_words SET enabled=1,level=?,category=?,note=?,updated_at=CURRENT_TIMESTAMP WHERE word=?", upd)
        ins = [(w["word"], w.get("level","high"), w.get("category",""), w.get("note","")) for w in words]
        c.executemany(
            "INSERT INTO forbidden_words(word,level,category,note,enabled) VALUES (?,?,?,?,1) ON CONFLICT DO NOTHING", ins)
        c.commit()
        print(f"--- [数据库] 批量写入 {len(words)} 条违禁词 ---")
    await _exec(_do)


async def load_forbidden_words_from_db() -> list[dict]:
    cursor = await _exec(lambda c: c.execute(
        "SELECT word,level,category FROM forbidden_words WHERE enabled=1"))
    return [{"word": r[0], "level": r[1], "category": r[2]} for r in await cursor.fetchall()]


async def get_recent_violations(limit: int = 50) -> list[dict]:
    cursor = await _exec(lambda c: c.execute(
        """SELECT id,room_id,asr_text,matched_word,risk_level,source,timestamp,
                  COALESCE(segment_id,'') AS segment_id
           FROM audit_log WHERE status='违规' ORDER BY timestamp DESC LIMIT ?""", (limit,)))
    rows = await cursor.fetchall()
    return [{"id": r[0], "room_id": r[1], "text": r[2], "word": r[3],
             "risk_level": r[4], "source": r[5], "time": r[6], "segment_id": r[7]} for r in rows]


async def get_room_stats(room_id: str = None) -> dict:
    where_clause = f"WHERE room_id={repr(room_id)}" if room_id else ""
    sql = f"""SELECT COUNT(*) as total,
              SUM(CASE WHEN status='违规' THEN 1 ELSE 0 END) as violations,
              SUM(CASE WHEN status='合规' THEN 1 ELSE 0 END) as compliant
             FROM audit_log {where_clause}"""
    cursor = await _exec(lambda c: c.execute(sql))
    row = await cursor.fetchone()
    return {"total": row[0] or 0, "violations": row[1] or 0,
            "compliant": row[2] or 0,
            "violation_rate": round(row[1] / max(row[0], 1) * 100, 2)}


async def get_active_sessions() -> list[dict]:
    cursor = await _exec(lambda c: c.execute(
        """SELECT s.session_id,s.room_id,s.client_ip,s.started_at,r.streamer
           FROM sessions s LEFT JOIN rooms r ON s.room_id=r.room_id
           WHERE s.status='active' ORDER BY s.started_at"""))
    rows = await cursor.fetchall()
    return [{"session_id": r[0], "room_id": r[1], "client_ip": r[2],
             "started_at": r[3], "streamer": r[4]} for r in rows]


# ===== low-level primitives used by save_session/end_session =====

async def _exec_commit(coro):
    """确保同一连接上执行 + commit。coro 是协程（如 lambda 返回 execute() 后，再 commit）。"""
    # 注意：这里需要用 acquire-release 模式保证同连接
    pool = await get_pool()
    conn = await pool.acquire()
    try:
        await coro
        conn.commit()
    finally:
        await pool.release(conn)
