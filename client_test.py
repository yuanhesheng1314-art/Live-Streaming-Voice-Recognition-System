# -*- coding: utf-8 -*-
"""
多路并发测试客户端
- 支持同时模拟多个直播间推送音频
- 自动处理心跳保活
- 实时展示审计结果

"""
import asyncio
import websockets
import json
import os
import re
import time
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Windows 下非 TTY 时 print 可能整块缓冲，审计结果看起来像「结束后才一起出来」
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

# WebSocket 基址；可用环境变量 WS_URI 覆盖（如连远程服务器）
_DEFAULT_WS_URI = "ws://127.0.0.1:8001/ws/audit"
FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_live.pcm")


def _try_load_client_auth_from_file() -> None:
    """
    可选本地密钥文件（避免每次 export）：
      test/.client_auth.env  或  项目根 .client_auth.env
    每行：AUTH_API_KEY=... 或 WS_AUTH_API_KEY=...（# 开头为注释）
    不覆盖已在环境中设置的值。
    """
    roots = [
        Path(__file__).resolve().parent / ".client_auth.env",
        Path(__file__).resolve().parent.parent / ".client_auth.env",
    ]
    for p in roots:
        if not p.is_file():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, _, v = s.partition("=")
                key = k.strip()
                if key not in ("AUTH_API_KEY", "WS_AUTH_API_KEY"):
                    continue
                val = v.strip().strip("'\"")
                if val:
                    os.environ.setdefault(key, val)
        except OSError:
            pass


_try_load_client_auth_from_file()


def build_ws_audit_uri(room_id: str) -> str:
    """
    拼接 /ws/audit 的 Query：room_id 必填；若服务端 AUTH_ENABLED=1，需带 api_key。
    客户端密钥来源（任选其一，与 systemd 里 AUTH_API_KEY 一致）：
      WS_AUTH_API_KEY  或  AUTH_API_KEY
    """
    base = (os.getenv("WS_URI") or _DEFAULT_WS_URI).strip()
    key = (os.getenv("WS_AUTH_API_KEY") or os.getenv("AUTH_API_KEY") or "").strip()
    parsed = urlparse(base)
    flat: dict[str, str] = {}
    for k, vlist in parse_qs(parsed.query, keep_blank_values=True).items():
        if vlist:
            flat[k] = vlist[0]
    flat["room_id"] = room_id
    if key:
        flat["api_key"] = key
    new_query = urlencode(flat)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )
# 是否打印 ASR 流式「转写」行。合并模式（默认）不在 transcript_final 单独打一行，避免与 audit 乱序。
# 默认关闭流式 partial：需要流式字幕时设 CLIENT_PRINT_PARTIAL=1。
# 需要看实时字幕式流式转写时：CLIENT_PRINT_PARTIAL=1 python client_test.py
PRINT_ASR_PARTIAL = os.getenv("CLIENT_PRINT_PARTIAL", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# 每条终稿与判定合并打印（避免终稿与 audit 分两截；默认开启）
COMBINED_SENTENCE_VERDICT = os.getenv(
    "CLIENT_COMBINED_SENTENCE_VERDICT", "1"
).strip().lower() not in ("0", "false", "no", "off")
# 音频发送完成后，若持续收不到新消息则自动结束，避免客户端长期阻塞等待。
RESULT_IDLE_TIMEOUT_SEC = float(os.getenv("CLIENT_RESULT_IDLE_TIMEOUT_SEC", "50"))
MAX_WAIT_AFTER_END_SEC = float(os.getenv("CLIENT_MAX_WAIT_AFTER_END_SEC", "150"))
# 距「上一次 audit」超过该秒数、且缓冲区里已有更大序号时，可跳过缺号（不受 heartbeat 干扰）。默认 0=不跳过。
COMBINED_END_SKIP_MISSING_SEC = float(
    os.getenv("CLIENT_COMBINED_END_SKIP_MISSING_SEC", "20")
)
CLIENT_PRINT_HIT_WORDS = os.getenv("CLIENT_PRINT_HIT_WORDS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CLIENT_PRINT_DEBUG_META = os.getenv("CLIENT_PRINT_DEBUG_META", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# 是否打印服务端快判（audit_fast）。终审（audit/audit_final）仍是最终统计口径。
CLIENT_PRINT_FAST_AUDIT = os.getenv("CLIENT_PRINT_FAST_AUDIT", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# 合并模式下是否仍打印 transcript_final（用于“开局有反馈”）；默认开启。
CLIENT_PRINT_TRANSCRIPT_FINAL = os.getenv(
    "CLIENT_PRINT_TRANSCRIPT_FINAL", "1"
).strip().lower() in ("1", "true", "yes", "on")
# 推流节奏：每帧时长（毫秒，16k/mono/s16le）。默认 80ms 略快于 100ms；过小可能影响云端端点
_CLIENT_ASR_FRAME_MS = float(os.getenv("CLIENT_ASR_FRAME_MS", "65"))
_CLIENT_CHUNK_BYTES = max(640, int(16000 * 2 * (_CLIENT_ASR_FRAME_MS / 1000.0)))
_CLIENT_SEND_SLEEP_SEC = float(
    os.getenv("CLIENT_SEND_SLEEP_SEC", str(_CLIENT_ASR_FRAME_MS / 1000.0))
)
# 音频发完后、发 end 前短暂等待，便于 ASR 消化尾帧（秒）
_CLIENT_END_TAIL_SLEEP_SEC = float(os.getenv("CLIENT_END_TAIL_SLEEP_SEC", "1.2"))


def ts():
    return datetime.now().strftime("%H:%M:%S")


def _short_reason(reason: str, max_len: int = 72) -> str:
    if not reason:
        return ""
    s = str(reason).strip().replace("\n", " ")
    for prefix in (
        "[LLM]",
        "[词库]",
        "[变体检测]",
        "[系统]",
        "[快速规则]",
        "[缓存复用]",
        "[系统繁忙]",
        "智能审计:",
    ):
        s = s.replace(prefix, "").strip()
    s = re.sub(r"^\[降级[^\]]*\]\s*", "", s).strip()
    if len(s) > max_len:
        return s[:max_len].rstrip() + "..."
    return s


def _seg_num(seg: str) -> int:
    """segment_id: <sid>-t000123 -> 123；异常返回大数（放到后面）。"""
    if not seg:
        return 10**9
    m = re.search(r"-t(\d+)$", str(seg))
    if not m:
        return 10**9
    try:
        return int(m.group(1))
    except ValueError:
        return 10**9


def _judgment_word(status: str) -> str:
    # 终端展示按用户要求统一为「合规/违规」
    if status in ("违禁", "违规"):
        return "违规"
    if status == "合规":
        return "合规"
    return "违规"


def _print_transcript_and_verdict(
    body: str,
    status: str,
    short: str,
    hits: list,
    *,
    risk: str = "",
    source: str = "",
    disorder_note: str = "",
    asr_ms: float | None = None,
    queue_ms: float | None = None,
    sem_wait_ms: float | None = None,
    infer_ms: float | None = None,
    llm_ms: float | None = None,
    total_ms: float | None = None,
) -> None:
    """终端：先转写句，再一行判定（合规/违规 + 简短理由）。"""
    body = body if len(body) <= 220 else (body[:220] + "…")
    j = _judgment_word(status)
    tail = ""
    if short:
        tail = f"，{short}"
    hit_s = ""
    if CLIENT_PRINT_HIT_WORDS and hits and status in ("违规", "违禁"):
        hit_s = f"（参考词：{','.join(str(x) for x in hits[:8])}）"
    meta = ""
    if CLIENT_PRINT_DEBUG_META and (risk or source):
        meta = f" [debug risk={risk or '-'} source={source or '-'}]"
    timing = ""
    if asr_ms is not None or llm_ms is not None:
        p = []
        if asr_ms is not None:
            p.append(f"ASR≈{asr_ms:.0f}ms")
        if queue_ms is not None:
            p.append(f"Q≈{queue_ms:.0f}ms")
        if sem_wait_ms is not None:
            p.append(f"SEM≈{sem_wait_ms:.0f}ms")
        if infer_ms is not None:
            p.append(f"INF≈{infer_ms:.0f}ms")
        if llm_ms is not None:
            p.append(f"LLM≈{llm_ms:.0f}ms")
        if total_ms is not None:
            p.append(f"T≈{total_ms:.0f}ms")
        timing = f" ({', '.join(p)})"
    note = f" {disorder_note}" if disorder_note else ""
    print(
        f"\n语音转写：{body}\n超管判断：{j}{tail}{hit_s}{timing}{meta}{note}",
        flush=True,
    )


async def send_heartbeat(ws, room_id: str):
    """定期发送心跳，保持连接"""
    try:
        while True:
            await asyncio.sleep(10)
            await ws.send(json.dumps({"type": "pong", "room_id": room_id}))
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        pass


async def _connect_ws(uri: str, room_id: str):
    """
    连接参数逐步降级。
    优先关闭客户端自动 ping：服务端忙于多句 Ollama 审计时，事件循环若短暂无法及时回 pong，
    易触发「keepalive ping timeout」(1011)，表现为只收到前几句 audit、后面全漏。
    """
    attempts = (
        dict(
            additional_headers={"X-Room-ID": room_id},
            ping_interval=None,
            ping_timeout=None,
            close_timeout=30,
        ),
        dict(
            extra_headers={"X-Room-ID": room_id},
            ping_interval=None,
            ping_timeout=None,
            close_timeout=30,
        ),
        dict(
            additional_headers={"X-Room-ID": room_id},
            ping_interval=30,
            ping_timeout=3600,
            close_timeout=10,
        ),
        dict(
            extra_headers={"X-Room-ID": room_id},
            ping_interval=30,
            ping_timeout=3600,
            close_timeout=10,
        ),
        dict(additional_headers={"X-Room-ID": room_id}),
        dict(extra_headers={"X-Room-ID": room_id}),
    )
    last_err = None
    for kw in attempts:
        try:
            return await websockets.connect(uri, **kw)
        except TypeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("websockets.connect: no attempts")


async def run_single_test(room_id: str, streamer: str = "测试主播"):
    """单路测试"""
    print(f"[{ts()}] [{room_id}] 正在连接 {streamer}...", flush=True)

    if not os.path.exists(FILE_PATH):
        print(f"  错误: 找不到音频文件 '{FILE_PATH}'", flush=True)
        print(f"  提示: 请在 data/ 目录下放置 16000Hz, 16bit, 单声道 PCM 文件", flush=True)
        return

    try:
        uri = build_ws_audit_uri(room_id)
        auth_key = (os.getenv("WS_AUTH_API_KEY") or os.getenv("AUTH_API_KEY") or "").strip()
        if not auth_key:
            print(
                "  提示: 未设置 WS_AUTH_API_KEY / AUTH_API_KEY；"
                "若服务端开启 AUTH_ENABLED=1 将出现 auth_failed。",
                flush=True,
            )
        ws = await _connect_ws(uri, room_id)
        async with ws:
            print(f"[{ts()}] [{room_id}] 已连接，开始推送音频...", flush=True)

            # 1. 发送元数据
            meta = {
                "type": "meta",
                "sample_rate": 16000,
                "room_id": room_id,
                "streamer": streamer
            }
            await ws.send(json.dumps(meta))

            # 启动心跳
            hb_task = asyncio.create_task(send_heartbeat(ws, room_id))

            violation_count = 0
            total_count = 0
            audit_sentence_count = 0
            transcript_sentence_count = 0
            audio_send_done = False
            audio_send_done_at = 0.0
            # audit 可能乱序到达：按 segment 序号缓冲后顺序打印（仅依赖 audit，避免与 transcript_final 交叉乱序）
            pending_audits: dict[int, dict] = {}
            transcript_recv_ts: dict[int, float] = {}
            transcript_asr_gap_ms: dict[int, float] = {}
            next_print_seq = 1

            try:
                # 并发推送音频 + 接收结果
                async def send_audio():
                    nonlocal total_count, audio_send_done, audio_send_done_at
                    with open(FILE_PATH, "rb") as f:
                        while True:
                            chunk = f.read(_CLIENT_CHUNK_BYTES)
                            if not chunk:
                                break
                            await ws.send(chunk)
                            total_count += 1
                            await asyncio.sleep(_CLIENT_SEND_SLEEP_SEC)
                    print(f"[{ts()}] [{room_id}] 音频推送完毕", flush=True)
                    # 给 ASR 一点时间消化最后一帧，再发 end，避免服务端过早收尾导致无 audit
                    await asyncio.sleep(_CLIENT_END_TAIL_SLEEP_SEC)
                    try:
                        await ws.send(json.dumps({"type": "end", "room_id": room_id}))
                    except Exception:
                        pass
                    audio_send_done = True
                    audio_send_done_at = time.time()

                async def receive_results():
                    nonlocal violation_count, audit_sentence_count, transcript_sentence_count, next_print_seq
                    last_msg_at = time.time()
                    last_audit_msg_at = time.time()
                    last_transcript_at = 0.0

                    def try_flush_audit_queue() -> None:
                        nonlocal next_print_seq, violation_count
                        while next_print_seq in pending_audits:
                            item = pending_audits.pop(next_print_seq)
                            # 先看到 transcript_final 再打印终审，避免“LLM 先于 ASR”观感
                            if next_print_seq not in transcript_recv_ts:
                                recv_at = float(item.get("audit_recv_ts", 0.0) or 0.0)
                                if recv_at <= 0 or (time.time() - recv_at) < 1.5:
                                    pending_audits[next_print_seq] = item
                                    break
                            body = item["text"]
                            st = item["status"]
                            if st in ("违规", "违禁"):
                                violation_count += 1
                            short = item["short"] or ""
                            _print_transcript_and_verdict(
                                body,
                                st,
                                short,
                                item["hits"],
                                risk=item["risk"],
                                source=item["source"],
                                asr_ms=item.get("asr_ms"),
                                    queue_ms=item.get("queue_ms"),
                                    sem_wait_ms=item.get("sem_wait_ms"),
                                    infer_ms=item.get("infer_ms"),
                                llm_ms=item.get("llm_ms"),
                                    total_ms=item.get("total_ms"),
                            )
                            next_print_seq += 1

                    while True:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            data = json.loads(message)
                            msg_type = data.get("type")
                            last_msg_at = time.time()

                            if msg_type == "partial":
                                # ASR 流式中间结果（与终稿可能重复；真实产品里常只做 UI 预览）
                                if PRINT_ASR_PARTIAL:
                                    text = data.get("text", "")
                                    if text:
                                        print(f"  [{room_id}] 转写: {text}", flush=True)

                            elif msg_type == "transcript_final":
                                # ASR 终稿：合并模式下不单独打印，避免与 audit 完成顺序不一致导致乱序/「审核中」假状态
                                seg = data.get("segment_id", "")
                                text = data.get("text", "")
                                s_no = _seg_num(seg)
                                if s_no < 10**9:
                                    now_ts = time.time()
                                    transcript_recv_ts[s_no] = now_ts
                                    if last_transcript_at > 0:
                                        transcript_asr_gap_ms[s_no] = (now_ts - last_transcript_at) * 1000.0
                                    else:
                                        transcript_asr_gap_ms[s_no] = 0.0
                                    last_transcript_at = now_ts
                                    transcript_sentence_count += 1
                                if text and ((not COMBINED_SENTENCE_VERDICT) or CLIENT_PRINT_TRANSCRIPT_FINAL):
                                    tag = f"  [{room_id}] 终稿[{seg}]: {text}"
                                    if COMBINED_SENTENCE_VERDICT:
                                        tag += "（审核中）"
                                    print(tag, flush=True)

                            elif msg_type == "audit_fast":
                                if CLIENT_PRINT_FAST_AUDIT:
                                    status = data.get("status", "未知")
                                    text = data.get("text", "")
                                    reason = _short_reason(data.get("reason", ""))
                                    source = str(data.get("source", "") or "").strip()
                                    perf = data.get("perf") or {}
                                    q = perf.get("queue_ms") if isinstance(perf, dict) else None
                                    tms = perf.get("total_ms") if isinstance(perf, dict) else None
                                    timing = ""
                                    p = []
                                    if isinstance(q, (int, float)):
                                        p.append(f"Q≈{float(q):.0f}ms")
                                    if isinstance(tms, (int, float)):
                                        p.append(f"T≈{float(tms):.0f}ms")
                                    if p:
                                        timing = f" ({', '.join(p)})"
                                    if text:
                                        if source == "queue_ack":
                                            print(
                                                f"\n语音转写：{text}\n快判结果：排队确认，结果稍后更新{timing}",
                                                flush=True,
                                            )
                                            continue
                                        print(
                                            f"\n语音转写：{text}\n快判结果：{_judgment_word(status)}"
                                            f"{('，' + reason) if reason else ''}{timing}",
                                            flush=True,
                                        )

                            elif msg_type in ("audit", "audit_final"):
                                # 审计结果（可与上句终稿合并为一组输出）
                                seg = data.get("segment_id", "")
                                seg_tag = f"[{seg}]" if seg else ""
                                status = data.get("status", "未知")
                                text = data.get("text", "")
                                reason = data.get("reason", "")
                                risk = data.get("risk_level", "") or data.get("level", "")
                                hits = data.get("hit_words") or []
                                source = str(data.get("source", "") or "").strip()
                                audit_sentence_count += 1
                                short = _short_reason(reason)
                                last_audit_msg_at = time.time()

                                if COMBINED_SENTENCE_VERDICT:
                                    s_no = _seg_num(seg)
                                    now_ts = time.time()
                                    llm_ms = None
                                    asr_ms = None
                                    queue_ms = None
                                    sem_wait_ms = None
                                    infer_ms = None
                                    total_ms = None
                                    perf = data.get("perf") or {}
                                    if isinstance(perf, dict):
                                        pm = perf.get("llm_ms")
                                        if isinstance(pm, (int, float)):
                                            llm_ms = float(pm)
                                        qm = perf.get("queue_ms")
                                        if isinstance(qm, (int, float)):
                                            queue_ms = float(qm)
                                        sm = perf.get("sem_wait_ms")
                                        if isinstance(sm, (int, float)):
                                            sem_wait_ms = float(sm)
                                        im = perf.get("infer_ms")
                                        if isinstance(im, (int, float)):
                                            infer_ms = float(im)
                                        tm = perf.get("total_ms")
                                        if isinstance(tm, (int, float)):
                                            total_ms = float(tm)
                                    if total_ms is None:
                                        if (queue_ms is not None) and (llm_ms is not None):
                                            total_ms = queue_ms + llm_ms
                                        elif queue_ms is not None:
                                            total_ms = queue_ms
                                        elif llm_ms is not None:
                                            total_ms = llm_ms
                                    t0 = transcript_recv_ts.get(s_no)
                                    if t0 is not None and llm_ms is None:
                                        llm_ms = (now_ts - t0) * 1000.0
                                    if s_no in transcript_asr_gap_ms:
                                        asr_ms = transcript_asr_gap_ms[s_no]
                                    pending_audits[s_no] = {
                                        "status": status,
                                        "text": text,
                                        "short": short,
                                        "risk": risk,
                                        "hits": hits,
                                        "source": source,
                                        "asr_ms": asr_ms,
                                        "queue_ms": queue_ms,
                                        "sem_wait_ms": sem_wait_ms,
                                        "infer_ms": infer_ms,
                                        "llm_ms": llm_ms,
                                        "total_ms": total_ms,
                                        "audit_recv_ts": now_ts,
                                    }
                                    try_flush_audit_queue()
                                    continue

                                seg_prefix = f"{seg_tag} " if seg_tag else ""
                                if status in ("违规", "违禁"):
                                    violation_count += 1
                                    tag = "【违禁】" if status == "违禁" else "【违规】"
                                    hit_hint = f" 参考词: {','.join(hits[:12])}" if hits else ""
                                    if short:
                                        print(
                                            f"\n  [{room_id}] {seg_prefix}{tag}{text}\n           理由: {short}  [{risk}]{hit_hint}",
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            f"\n  [{room_id}] {seg_prefix}{tag}{text}  [{risk}]{hit_hint}",
                                            flush=True,
                                        )
                                elif status == "合规":
                                    if short:
                                        print(
                                            f"  [{room_id}] {seg_prefix}【合规】{text}（{short}）",
                                            flush=True,
                                        )
                                    else:
                                        print(f"  [{room_id}] {seg_prefix}【合规】{text}", flush=True)
                                else:
                                    if short:
                                        print(
                                            f"  [{room_id}] {seg_prefix}【{status}】{text}（{short}）",
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            f"  [{room_id}] {seg_prefix}【{status}】{text}",
                                            flush=True,
                                        )

                            elif msg_type == "heartbeat":
                                # 服务端心跳，回应 pong
                                await ws.send(json.dumps({"type": "pong", "room_id": room_id}))

                            elif msg_type == "error":
                                code = data.get("code", "")
                                msg = data.get("message", "")
                                print(
                                    f"[{ts()}] [{room_id}] 服务端错误"
                                    f"{(' [' + str(code) + ']') if code else ''}: {msg}",
                                    flush=True,
                                )

                        except websockets.exceptions.ConnectionClosed as closed:
                            print(
                                f"[{ts()}] [{room_id}] 连接已关闭 "
                                f"(code={getattr(closed, 'code', '?')}, "
                                f"reason={getattr(closed, 'reason', '')!r})",
                                flush=True,
                            )
                            break
                        except asyncio.TimeoutError:
                            now = time.time()
                            if (
                                COMBINED_SENTENCE_VERDICT
                                and pending_audits
                                and audio_send_done
                                and COMBINED_END_SKIP_MISSING_SEC > 0
                            ):
                                min_k = min(pending_audits.keys())
                                if (
                                    min_k > next_print_seq
                                    and (now - last_audit_msg_at)
                                    >= COMBINED_END_SKIP_MISSING_SEC
                                ):
                                    for s in range(next_print_seq, min_k):
                                        print(
                                            f"[{ts()}] [{room_id}] 提示：序号 {s} 未收到审计结果，"
                                            f"已跳过（>{COMBINED_END_SKIP_MISSING_SEC:.0f}s 无新 audit）。",
                                            flush=True,
                                        )
                                    next_print_seq = min_k
                                    try_flush_audit_queue()
                            if audio_send_done:
                                idle = now - last_msg_at
                                waited = now - audio_send_done_at
                                pending_expected = max(0, transcript_sentence_count - audit_sentence_count)
                                if waited >= MAX_WAIT_AFTER_END_SEC or (
                                    idle >= RESULT_IDLE_TIMEOUT_SEC and pending_expected <= 0
                                ):
                                    print(
                                        f"[{ts()}] [{room_id}] 等待审计结果超时，结束本次测试 "
                                        f"(idle={idle:.1f}s, waited={waited:.1f}s, pending={pending_expected})",
                                        flush=True,
                                    )
                                    break
                            continue
                        except Exception as e:
                            print(f"[{ts()}] [{room_id}] 接收异常: {e}", flush=True)
                            break

                await asyncio.gather(send_audio(), receive_results())

            finally:
                hb_task.cancel()

            if COMBINED_SENTENCE_VERDICT and pending_audits:
                ks = sorted(pending_audits.keys())
                head = ",".join(str(x) for x in ks[:8])
                more = "…" if len(ks) > 8 else ""
                print(
                    f"[{ts()}] [{room_id}] 提示：连接已结束时仍有 {len(pending_audits)} 条 audit "
                    f"未按序打印（当前等待序号 {next_print_seq}，缓冲序号：{head}{more}）。",
                    flush=True,
                )

            print(
                f"[{ts()}] [{room_id}] 测试完成 - 总帧: {total_count}, "
                f"ASR终稿句数: {transcript_sentence_count}, 审计句数: {audit_sentence_count}, 违规: {violation_count}",
                flush=True,
            )

    except Exception as e:
        print(f"[{ts()}] [{room_id}] 连接失败: {e}", flush=True)


async def run_multi_test(rooms: list[dict]):
    """
    多路并发测试
    rooms: [{"room_id": "room_001", "streamer": "主播A"}, ...]
    """
    print("=" * 60)
    print(f"  多路并发测试 - {len(rooms)} 个直播间")
    print("=" * 60)

    tasks = []
    for room in rooms:
        tasks.append(run_single_test(room["room_id"], room["streamer"]))

    await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print("  全部测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 单路测试模式
    if len(sys.argv) < 2 or sys.argv[1] == "single":
        asyncio.run(run_single_test("test_room_001", "测试主播A"))

    # 多路测试模式: python client_test.py multi
    elif sys.argv[1] == "multi":
        rooms = [
            {"room_id": "room_001", "streamer": "主播A"},
            {"room_id": "room_002", "streamer": "主播B"},
            {"room_id": "room_003", "streamer": "主播C"},
            {"room_id": "room_004", "streamer": "主播D"},
        ]
        asyncio.run(run_multi_test(rooms))
