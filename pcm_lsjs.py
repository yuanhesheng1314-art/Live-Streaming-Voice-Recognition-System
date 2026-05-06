# -*- coding: utf-8 -*-
"""
直播间语音识别与违禁词审计系统 v2.0
- 多路并发音频流
- 违禁词热加载 + 分级管理
- 谐音/变体检测
- 审计日志持久化 (SQLite)
- WebSocket 心跳保活
- REST 管理 API
"""
import os
import uuid
import base64
import asyncio
import json
import re
import time
import threading
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import unicodedata
import traceback
import logging

import httpx
import numpy as np
import ahocorasick
import uvicorn
import dashscope
from scipy import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from dashscope.audio.qwen_omni import OmniRealtimeConversation, OmniRealtimeCallback, MultiModality
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

# 本地模块
from db import (
    init_db, save_session, end_session, log_audit,
    upsert_forbidden_words, load_forbidden_words_from_db,
    get_recent_violations, get_room_stats, get_active_sessions
)
from variant_detect import VariantDetector

try:
    from variant_detect import generate_variants
except ImportError:
    from variant_detect import generate_variants_cached as generate_variants

# ================= 配置管理 =================
load_dotenv(dotenv_path="apikey.env")
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 项目根目录（src）与仓库根目录（常见把 yjc.txt 放在 live_audit/ 下）
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
# 实际加载路径在 load_yjc_file 中解析后写入 YJC_ACTIVE_PATH
YJC_PATH = PROJECT_DIR / "yjc.txt"
YJC_ACTIVE_PATH: Path = YJC_PATH


class AppConfig:
    # Ollama LLM 配置
    OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
    OLLAMA_GENERATE_URL = os.getenv("OLLAMA_GENERATE_URL", "http://127.0.0.1:11434/api/generate")
    # 语言模型终审（本机 Ollama）：精准率优先，8B模型平衡速度与精准率
    # qwen3:8b（主要模型）+ qwen2.5:3b（备用模型）
    _ollama_model_raw = (os.getenv("OLLAMA_MODEL", "qwen3:8b") or "qwen3:8b").strip()
    OLLAMA_MODEL = (
        _ollama_model_raw.replace("qwen3-8b", "qwen3:8b")
        .replace("Qwen3-8B", "qwen3:8b")
        .replace("Qwen3:8B", "qwen3:8b")
        .replace("Qwen3-14B", "qwen3:14b")
        .replace("Qwen3:32B", "qwen3:32b")
    )
    # 备用模型（当主模型并发不足时使用）
    OLLAMA_MODEL_FALLBACK = os.getenv("OLLAMA_MODEL_FALLBACK", "qwen2.5:3b")
    # 精准率优先：快判也使用主模型，避免小模型导致的误判
    _fast_raw = (os.getenv("OLLAMA_MODEL_FAST", "") or "").strip()
    # 精准率优先：快判默认使用主模型，避免小模型导致的误判
    LLM_USE_FAST_MODEL = os.getenv("LLM_USE_FAST_MODEL", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 快判路径默认适中上下文；终审由 full prompt 保证准确率
    LLM_FAST_NUM_CTX = max(256, int(os.getenv("LLM_FAST_NUM_CTX", "768")))
    LLM_FAST_NUM_PREDICT = max(64, int(os.getenv("LLM_FAST_NUM_PREDICT", "192")))
    # A100 优化：增加超时以获得更好的推理质量
    OLLAMA_HTTP_READ_TIMEOUT = float(os.getenv("OLLAMA_HTTP_READ_TIMEOUT", "120"))
    # A100优化：降低全局并发上限，避免单路过长影响整体
    OLLAMA_MAX_CONCURRENT = max(1, int(os.getenv("OLLAMA_MAX_CONCURRENT", "4")))
    # A100 可并发更高，但过大并发会拉高单句时延并触发固定超时；默认取稳态值
    _ogg = (os.getenv("OLLAMA_GLOBAL_MAX_CONCURRENT", "") or "").strip()
    OLLAMA_GLOBAL_MAX_CONCURRENT = max(1, int(_ogg)) if _ogg else 6
    # 终审输出预算：保证准确率同时抑制长尾
    OLLAMA_NUM_PREDICT = max(96, int(os.getenv("OLLAMA_NUM_PREDICT", "256")))
    # think=true 时模型先输出思考再写 JSON，需更大 num_predict 以免正文 JSON 被截断
    OLLAMA_NUM_PREDICT_WITH_THINK = max(256, int(os.getenv("OLLAMA_NUM_PREDICT_WITH_THINK", "1024")))
    # 分类稳定性：略压 top_p、略抬 repeat_penalty，减少「模棱两可」与重复车轱辘话（可用环境变量微调）
    OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.88"))
    OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.12"))
    # 主请求解析不出 JSON 时是否再发极简 chat（多一次 HTTP；追求速度可设 0）
    LLM_PARSE_RETRY = int(os.getenv("LLM_PARSE_RETRY", "0"))
    # 单句终审预算（准确率优先）：默认 45 秒，减少长句被硬超时
    LLM_SINGLE_TIMEOUT_SEC = float(os.getenv("LLM_SINGLE_TIMEOUT_SEC", "45"))
    # 等待全局并发槽位预算（秒）
    LLM_SEM_WAIT_TIMEOUT_SEC = float(os.getenv("LLM_SEM_WAIT_TIMEOUT_SEC", "20"))
    # 快判推理预算（秒）
    LLM_FAST_TIMEOUT_SEC = float(os.getenv("LLM_FAST_TIMEOUT_SEC", "28"))
    # 精准率优先：单次完整终审，避免「快判超时 + 复核再超时」双段丢模与词库误降级
    LLM_AUDIT_SINGLE_CALL = os.getenv("LLM_AUDIT_SINGLE_CALL", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # 单句终审超时后是否自动做一次紧凑重试（缩小上下文和输出预算）
    LLM_TIMEOUT_RETRY_ONCE = os.getenv("LLM_TIMEOUT_RETRY_ONCE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # 紧凑重试额外预算（秒）
    LLM_TIMEOUT_RETRY_EXTRA_SEC = float(os.getenv("LLM_TIMEOUT_RETRY_EXTRA_SEC", "18"))
    # 提速开关：使用更短的终审提示词，降低首 token 与总解码时延
    LLM_FAST_PROMPT = os.getenv("LLM_FAST_PROMPT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 提速开关：高置信规则直判（减少 LLM 调用量）
    # 默认关闭「快速规则优先直判」：优先走 LLM 终审提升准确率；拥塞时可再开启兜底提速
    LLM_ENABLE_RULE_SHORTCUT = os.getenv("LLM_ENABLE_RULE_SHORTCUT", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # chat/generate 遇 ReadTimeout、连接重置等瞬态错误时的额外重试次数（不含首次，默认 2 即最多 3 次请求）
    OLLAMA_TRANSIENT_RETRIES = max(0, int(os.getenv("OLLAMA_TRANSIENT_RETRIES", "1")))
    # 使用 Ollama JSON Schema 约束 verdict（新 Ollama 支持，可显著减少「非法 verdict」与漏解析）
    OLLAMA_USE_JSON_SCHEMA = os.getenv("OLLAMA_USE_JSON_SCHEMA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 终审上下文：默认 1536，长句准确率与时延较平衡
    LLM_AUDIT_NUM_CTX = max(512, int(os.getenv("LLM_AUDIT_NUM_CTX", "1536")))
    # 单句终审携带的历史条数：平衡准确率与速度
    LLM_CONTEXT_TURNS = max(1, int(os.getenv("LLM_CONTEXT_TURNS", "5")))
    # 模型返回「合规」但命中高风险模式时，启用快速复核，减少漏判
    LLM_ENABLE_SUSPECT_RECHECK = os.getenv("LLM_ENABLE_SUSPECT_RECHECK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    LLM_RECHECK_TIMEOUT_SEC = float(os.getenv("LLM_RECHECK_TIMEOUT_SEC", "15"))
    # 精准率优先：关闭高置信规则预筛，让LLM终审所有句子
    LLM_ENABLE_HIGHCONF_RULE_PREFILTER = os.getenv(
        "LLM_ENABLE_HIGHCONF_RULE_PREFILTER", "0"
    ).strip().lower() not in ("0", "false", "no", "off")
    # 精准率优先：关闭高置信合规预筛，让LLM终审所有句子
    LLM_ENABLE_BENIGN_PREFILTER = os.getenv(
        "LLM_ENABLE_BENIGN_PREFILTER", "0"
    ).strip().lower() not in ("0", "false", "no", "off")
    # 低信息短句（如“嗯/好的/收到”）直接给合规，避免占用终审队列造成长尾超时。
    LLM_SKIP_LOWINFO_UTTERANCE = os.getenv("LLM_SKIP_LOWINFO_UTTERANCE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # 入队后先回一条 audit_fast（排队确认），减少客户端“>20s 无审计”误判。
    AUDIT_SEND_QUEUE_ACK = os.getenv("AUDIT_SEND_QUEUE_ACK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # 随机模型回退会带来不稳定准确率；默认关闭，仅在明确压测吞吐时开启。
    LLM_ENABLE_RANDOM_MODEL_FALLBACK = os.getenv(
        "LLM_ENABLE_RANDOM_MODEL_FALLBACK", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    # 增加收尾等待时间，确保所有审计完成
    AUDIT_DRAIN_TIMEOUT = float(os.getenv("AUDIT_DRAIN_TIMEOUT", "600"))

    # 音频处理
    TARGET_RATE = 16000
    # 同时进行中的 WebSocket 直播路数（活跃会话）硬上限
    MAX_CONCURRENT = max(1, int(os.getenv("WS_MAX_SESSIONS", "8")))
    # 上下文窗口上限（用于历史裁剪）
    DYNAMIC_CTX = 2048

    # 连接管理（长音频 + 多句 LLM 时，客户端可能数十秒才发下一条；过小会误断）
    AUDIO_TIMEOUT = int(os.getenv("WS_AUDIO_RECEIVE_TIMEOUT", "120"))
    HEARTBEAT_INTERVAL = 15  # 心跳间隔（秒）
    HEARTBEAT_TIMEOUT = 45  # 心跳超时（秒），超过此时间无数据视为断连

    # DashScope 实时 ASR 断链重连与熔断（与 ws/audit 中逻辑一致，缺省会直接 AttributeError）
    ASR_UPSTREAM_RECONNECT_RETRIES = max(
        0, int(os.getenv("ASR_UPSTREAM_RECONNECT_RETRIES", "2"))
    )
    ASR_UPSTREAM_RECONNECT_BACKOFF_SEC = float(
        os.getenv("ASR_UPSTREAM_RECONNECT_BACKOFF_SEC", "0.6")
    )
    ASR_UPSTREAM_RECONNECT_COOLDOWN_SEC = float(
        os.getenv("ASR_UPSTREAM_RECONNECT_COOLDOWN_SEC", "1.2")
    )
    ASR_UPSTREAM_FATAL_FAIL_WINDOW_SEC = float(
        os.getenv("ASR_UPSTREAM_FATAL_FAIL_WINDOW_SEC", "15")
    )
    ASR_UPSTREAM_FATAL_FAIL_THRESHOLD = max(
        1, int(os.getenv("ASR_UPSTREAM_FATAL_FAIL_THRESHOLD", "8"))
    )
    ASR_UPSTREAM_PAUSE_SEC = float(os.getenv("ASR_UPSTREAM_PAUSE_SEC", "6"))
    # 百炼「Qwen3-ASR-Flash」实时语音识别：WebSocket 模型 ID 为 qwen3-asr-flash-realtime（稳定名，指向当前快照）。
    # 离线/文件转写 qwen3-asr-flash 不走 OmniRealtimeConversation，与本项目直播链路不兼容。
    _dashscope_asr_model = (
        os.getenv("DASHSCOPE_ASR_MODEL", "qwen3-asr-flash-realtime") or ""
    ).strip()
    DASHSCOPE_ASR_MODEL = _dashscope_asr_model or "qwen3-asr-flash-realtime"
    _dashscope_asr_fb = (os.getenv("DASHSCOPE_ASR_MODEL_FALLBACK", "") or "").strip()
    DASHSCOPE_ASR_MODEL_FALLBACK = _dashscope_asr_fb
    # 实时 Omni/ASR WebSocket 基址（不要带 ?model=，SDK 会自动拼接）。
    # 默认 None 表示走 SDK 内置北京节点 wss://dashscope.aliyuncs.com/api-ws/v1/realtime
    # 国际（新加坡）控制台申请的 Key 通常必须设为：
    #   DASHSCOPE_REALTIME_WS_URL=wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime
    _dashscope_rt_ws = (os.getenv("DASHSCOPE_REALTIME_WS_URL", "") or "").strip()
    DASHSCOPE_REALTIME_WS_URL: Optional[str] = _dashscope_rt_ws if _dashscope_rt_ws else None
    _dashscope_ws = (os.getenv("DASHSCOPE_WORKSPACE_ID", "") or "").strip()
    DASHSCOPE_WORKSPACE_ID: Optional[str] = _dashscope_ws if _dashscope_ws else None
    # DashScope 上游长时间无音频可能 idle 断连：定时追加极短静音（与 apikey.env 中 ASR_KEEPALIVE_* 一致）
    ASR_KEEPALIVE_ENABLED = os.getenv("ASR_KEEPALIVE_ENABLED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    ASR_KEEPALIVE_INTERVAL_SEC = max(0.5, float(os.getenv("ASR_KEEPALIVE_INTERVAL_SEC", "4")))
    ASR_KEEPALIVE_PCM_MS = max(10, int(os.getenv("ASR_KEEPALIVE_PCM_MS", "100")))
    # connect() 只保证 TCP/WS 连通；须在收到 session.updated 后再 append_audio，否则易首包即断（见阿里云实时语音识别文档）
    ASR_SESSION_UPDATED_TIMEOUT_SEC = float(
        os.getenv("ASR_SESSION_UPDATED_TIMEOUT_SEC", "25")
    )
    # 0 / false / off → update_session(..., enable_turn_detection=False)，与文档 Manual 模式一致，部分网关对默认 server_vad 更挑剔时可试
    ASR_SERVER_VAD = os.getenv("ASR_SERVER_VAD", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # 队列深度加大，减少队列满导致的降级
    MAX_QUEUE_SIZE = 150  # 异步审计队列深度
    # 每个会话的审计 worker 数：A100 场景默认 2，降低长队列等待
    AUDIT_WORKERS_PER_SESSION = max(1, int(os.getenv("AUDIT_WORKERS_PER_SESSION", "2")))
    # 高优先级队列占比（命中高风险特征的句子走高优先）
    AUDIT_PRIORITY_QUEUE_RATIO = float(os.getenv("AUDIT_PRIORITY_QUEUE_RATIO", "0.3"))
    # 高优先级连续处理上限
    AUDIT_HI_BURST = max(1, int(os.getenv("AUDIT_HI_BURST", "5")))
    # 收尾阶段触发快速降级的排队阈值（毫秒）
    TAIL_FAST_QUEUE_WAIT_MS = float(os.getenv("TAIL_FAST_QUEUE_WAIT_MS", "30000"))
    # 常态排队预算
    AUDIT_QUEUE_BUDGET_MS = float(os.getenv("AUDIT_QUEUE_BUDGET_MS", "20000"))
    # 准确率优先：默认关闭排队预算直接降级，避免高 Q 时大量词库短路误判。
    ENABLE_QUEUE_BUDGET_DEGRADE = os.getenv(
        "ENABLE_QUEUE_BUDGET_DEGRADE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    # 准确率优先：默认关闭收尾强制降级，尽量给出 LLM 最终判定。
    ENABLE_TAIL_FAST_DEGRADE = os.getenv(
        "ENABLE_TAIL_FAST_DEGRADE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    # 高负载时是否降级/跳过变体检测，避免低价值检测占用吞吐
    ENABLE_VARIANT_UNDER_LOAD = os.getenv("ENABLE_VARIANT_UNDER_LOAD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 高风险词库前置直判（命中 high 且非劝导/辟谣语境时，优先实时拦截）
    ENABLE_HIGHRISK_WORDLIST_PREFILTER = os.getenv(
        "ENABLE_HIGHRISK_WORDLIST_PREFILTER", "1"
    ).strip().lower() not in ("0", "false", "no", "off")
    VARIANT_SKIP_BACKLOG = max(1, int(os.getenv("VARIANT_SKIP_BACKLOG", "10")))
    # 性能日志：输出句级排队/推理耗时，便于压测定位瓶颈
    AUDIT_PERF_LOG = os.getenv("AUDIT_PERF_LOG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 双阶段输出：先给 audit_fast（快判），再给 audit_final（终审）
    ENABLE_DUAL_STAGE_AUDIT = os.getenv("ENABLE_DUAL_STAGE_AUDIT", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    # 快判后是否做冲突复核（建议开启：不明显牺牲准确率）
    ENABLE_CONFLICT_RECHECK = os.getenv("ENABLE_CONFLICT_RECHECK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    FAST_RECHECK_CONFIDENCE = float(os.getenv("FAST_RECHECK_CONFIDENCE", "0.72"))
    # 动态重载保护：拥塞时自动提高复核阈值，降低 full 终审比例以稳定时延
    ENABLE_DYNAMIC_RECHECK_GUARD = os.getenv("ENABLE_DYNAMIC_RECHECK_GUARD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    RECHECK_GUARD_BACKLOG = max(1, int(os.getenv("RECHECK_GUARD_BACKLOG", "8")))
    RECHECK_GUARD_QUEUE_MS = float(os.getenv("RECHECK_GUARD_QUEUE_MS", "1500"))
    RECHECK_GUARD_BOOST = float(os.getenv("RECHECK_GUARD_BOOST", "0.15"))
    RECHECK_GUARD_BOOST_HEAVY = float(os.getenv("RECHECK_GUARD_BOOST_HEAVY", "0.08"))
    # 跨句组合规则：默认开启；误杀时可设 0 关闭
    ENABLE_CROSS_SENTENCE_RULE = os.getenv("ENABLE_CROSS_SENTENCE_RULE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    # 违禁词文件监控
    YJC_WATCH_INTERVAL = 2  # 文件变化检测间隔（秒）

    # 实时告警配置（支持钉钉/企业微信/自定义Webhook）
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")  # 告警推送地址
    ALERT_ENABLED = os.getenv("ALERT_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
    ALERT_MIN_RISK = os.getenv("ALERT_MIN_RISK", "high")  # 仅告警 high/medium/low 及以上
    # 认证配置
    AUTH_API_KEY = os.getenv("AUTH_API_KEY", "")  # 客户端认证密钥（空则不校验）
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    # 是否允许从 ?api_key= 读取客户端密钥（会出现在访问日志 URL 中；默认关闭，仅走 Authorization / X-API-Key）
    AUTH_WS_ALLOW_QUERY_KEY = os.getenv("AUTH_WS_ALLOW_QUERY_KEY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Ollama：终审输出 JSON Schema（显著降低非 JSON / verdict 非法概率，利于直播不漏判）
VERDICT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["合规", "违规", "违禁"]},
        "reason": {"type": "string"},
        "risk_level": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        },
    },
    "required": ["verdict", "reason", "risk_level"],
}


def ollama_verdict_format_sequence() -> list:
    """依次尝试的 chat format：Schema → legacy json → 不设 format。"""
    if not AppConfig.OLLAMA_USE_JSON_SCHEMA:
        return ["json", None]
    return [VERDICT_JSON_SCHEMA, "json", None]


# ================= 实时告警 =================

async def send_alert_webhook(
    room_id: str,
    text: str,
    status: str,
    risk_level: str,
    reason: str,
    matched_word: str = "",
    session_id: str = "",
):
    """发送实时告警到钉钉/企业微信/自定义Webhook"""
    if not AppConfig.ALERT_ENABLED or not AppConfig.ALERT_WEBHOOK_URL:
        return

    # 风险等级过滤
    risk_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
    min_risk = risk_order.get(AppConfig.ALERT_MIN_RISK, 3)
    current_risk = risk_order.get(risk_level, 0)
    if current_risk < min_risk:
        return

    # 仅告警违规/违禁
    if status not in ("违规", "违禁"):
        return

    try:
        alert_payload = {
            "msgtype": "text",
            "text": {
                "content": (
                    f"⚠️ 直播违规告警\n"
                    f"房间: {room_id}\n"
                    f"等级: {risk_level} | {status}\n"
                    f"原文: {text[:200]}\n"
                    f"命中词: {matched_word}\n"
                    f"原因: {reason[:100]}\n"
                    f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            }
        }
        # 钉钉/企微兼容格式
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                AppConfig.ALERT_WEBHOOK_URL,
                json=alert_payload,
                headers={"Content-Type": "application/json"},
            )
        print(f"--- [告警] 已推送 {room_id} {status} 告警 ---")
    except Exception as e:
        print(f"--- [告警] 推送失败: {e} ---")


def ollama_request_top_level_extras() -> dict:
    """
    Ollama /api/chat、/api/generate 顶层扩展字段（与 model、messages 同级，勿放进 options）。
    Qwen3：think=true 利于深度推理；本服务对 chat 返回会「优先从 message.content 解析 JSON」再回退合并串，降低思考与 JSON 混杂导致的解析失败。
    环境变量 OLLAMA_API_THINK：
      on（默认）→ think:true，配合加大的 OLLAMA_NUM_PREDICT_WITH_THINK；
      off → think:false，略省 token、适合旧版或仍解析不稳时；
      omit → 不传 think（极旧 Ollama 若报未知字段可设此项）。
    """
    raw = (os.getenv("OLLAMA_API_THINK", "off") or "off").strip().lower()
    if raw in ("omit", "skip", "none"):
        return {}
    if raw in ("1", "true", "yes", "on"):
        return {"think": True}
    return {"think": False}


def merge_ollama_chat_message_parts(msg: dict) -> str:
    """合并 Ollama chat 返回的 message 正文与 thinking/reasoning（用于日志与兜底解析）。"""
    parts: list[str] = []
    c = msg.get("content")
    if isinstance(c, str) and c.strip():
        parts.append(c.strip())
    for key in ("thinking", "reasoning"):
        t = msg.get(key)
        if isinstance(t, str) and t.strip():
            parts.append(t.strip())
    return "\n".join(parts) if parts else ""


def iter_verdict_parse_strings_from_chat(msg: dict):
    """JSON 解析候选：优先 message.content（think 分离时常仅此处为 verdict JSON），再尝试全文合并。"""
    c = msg.get("content")
    content = (c or "").strip() if isinstance(c, str) else ""
    merged = merge_ollama_chat_message_parts(msg)
    if content:
        yield content
    if merged and merged != content:
        yield merged
    elif not content and merged:
        yield merged


def ollama_audit_num_predict() -> int:
    """think=true 时自动抬高 num_predict，减少「思考占满 token、JSON 被截断」。"""
    ex = ollama_request_top_level_extras()
    if ex.get("think") is True:
        return max(AppConfig.OLLAMA_NUM_PREDICT, AppConfig.OLLAMA_NUM_PREDICT_WITH_THINK)
    return AppConfig.OLLAMA_NUM_PREDICT


def llm_chat_model_name(fast_mode: bool) -> str:
    """快判可走小模型；终审用 OLLAMA_MODEL（支持回退）。"""
    if fast_mode and AppConfig.LLM_USE_FAST_MODEL:
        m = (AppConfig.OLLAMA_MODEL_FAST or "").strip()
        if m:
            return m
    return AppConfig.OLLAMA_MODEL

def llm_fallback_model_name() -> str:
    """获取备用模型名称（用于回退）。"""
    return AppConfig.OLLAMA_MODEL_FALLBACK if AppConfig.OLLAMA_MODEL_FALLBACK else AppConfig.OLLAMA_MODEL


def should_use_fallback_model(text: str, segment_id: str) -> bool:
    """
    判断是否应该使用备用模型（当主模型并发不足时）。
    策略：根据当前全局并发使用情况和请求频率动态决定。
    """
    if not AppConfig.LLM_ENABLE_RANDOM_MODEL_FALLBACK:
        return False
    global _model_fallback_counter
    _model_fallback_counter = (_model_fallback_counter + 1) % 1000

    # 如果没有配置备用模型，不回退
    if not AppConfig.OLLAMA_MODEL_FALLBACK:
        return False

    # 主模型和备用模型相同，不回退
    if AppConfig.OLLAMA_MODEL_FALLBACK == AppConfig.OLLAMA_MODEL:
        return False

    # 检查是否该句子已经回退过（避免重复回退）
    fallback_key = f"{segment_id}"
    if fallback_key in _model_fallback_locks:
        return False

    # 随机回退策略（避免所有句子同时回退到备用模型）
    # 使用段落计数作为随机种子的一部分
    random_factor = _model_fallback_counter % 10
    if random_factor != 0:  # 90% 概率使用主模型
        return False

    return True


# 全局模型回退锁（避免同一句话多次回退）
_model_fallback_locks: dict[str, bool] = {}
_model_fallback_counter = 0


def llm_num_ctx_predict(
    fast_mode: bool,
    *,
    text_len: int = 0,
    history_turns: int = 0,
    force_compact: bool = False,
) -> tuple[int, int]:
    """返回 (num_ctx, num_predict)。准确率优先前提下做动态预算，降低长尾超时。"""
    if fast_mode:
        n_ctx = AppConfig.LLM_FAST_NUM_CTX
        n_pred = AppConfig.LLM_FAST_NUM_PREDICT
    else:
        n_ctx = AppConfig.LLM_AUDIT_NUM_CTX
        n_pred = ollama_audit_num_predict()

    if history_turns <= 0:
        n_ctx = min(n_ctx, 1024 if not fast_mode else 768)
    if text_len <= 120:
        n_ctx = min(n_ctx, 768 if not fast_mode else 512)
    elif text_len <= 260:
        n_ctx = min(n_ctx, 1024 if not fast_mode else 768)

    if force_compact:
        n_ctx = min(n_ctx, 768 if not fast_mode else 512)
        n_pred = min(n_pred, 160 if not fast_mode else 96)

    return int(max(256, n_ctx)), int(max(64, n_pred))


def llm_chat_top_level_extras(fast_mode: bool) -> dict:
    """快判强制关 think，避免小模型把 token 浪费在思考段。"""
    ex = dict(ollama_request_top_level_extras())
    if fast_mode:
        ex["think"] = False
    return ex


# ================= 全局组件 =================
ac_automaton = ahocorasick.Automaton()
variant_detector = VariantDetector()

# 词库为空时 pyahocorasick 仍要求至少 add_word 一次并完成 make_automaton，否则 iter() 会报错
AC_AUTOMATON_EMPTY_SENTINEL = "\ufdd0__AC_EMPTY__\ufdd1"

# 连接跟踪
active_sessions: dict[str, "LiveAuditManager"] = {}
# 新建 ASR 会话准入锁（与 MAX_CONCURRENT 配合，避免「同时首包」竞态超路）
_session_admission_lock = asyncio.Lock()
# 全进程 Ollama 终审并发（所有房间、所有句子叠加），保护单机 GPU/HTTP 不被冲垮
_global_ollama_sem = asyncio.Semaphore(max(1, AppConfig.OLLAMA_GLOBAL_MAX_CONCURRENT))

# Uvicorn/FastAPI 主事件循环（供 Watchdog 线程里安全调度异步重建词库）
_app_loop: Optional[asyncio.AbstractEventLoop] = None


# ================= 违禁词管理 =================

def _resolve_yjc_paths() -> list[Path]:
    """候选词库路径：环境变量 > src/yjc.txt > 项目根 yjc.txt"""
    paths: list[Path] = []
    env_p = os.getenv("YJC_FILE") or os.getenv("YJC_PATH")
    if env_p:
        paths.append(Path(env_p).expanduser().resolve())
    paths.append((PROJECT_DIR / "yjc.txt").resolve())
    paths.append((REPO_ROOT / "yjc.txt").resolve())
    # 去重保序
    seen = set()
    out: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _parse_pipe_line(line: str) -> list[dict]:
    """传统：关键词|等级|分类|备注"""
    parts = line.split("|")
    word = parts[0].strip()
    if not word or word.startswith("#"):
        return []
    level = parts[1].strip() if len(parts) > 1 else "high"
    category = parts[2].strip() if len(parts) > 2 else ""
    note = parts[3].strip() if len(parts) > 3 else ""
    if level not in ("high", "medium", "low"):
        level = "high"
    return [{"word": word, "level": level, "category": category, "note": note}]


def _split_zh_terms(blob: str) -> list[str]:
    """从「国家级、最高级、最佳…」拆出词条"""
    blob = blob.strip()
    if not blob:
        return []
    # 统一分隔：顿号、逗号、分号、空白
    blob = blob.replace("，", ",").replace("；", ",").replace("、", ",")
    raw = [t.strip() for t in blob.split(",")]
    out: list[str] = []
    for t in raw:
        if not t:
            continue
        t = re.sub(r"\.{2,}$", "", t).strip()  # 去掉行尾 ...
        if len(t) >= 1 and t not in ("…", "..."):
            out.append(t)
    return out


def _parse_doc_style_lines(lines: list[str]) -> list[dict]:
    """
    解析「直播违禁词表」文档体例，例如：
      一、绝对化极限词
      •最高级/排位类：国家级、最高级、最佳、...
      三、合规替代话术推荐
      •极限词替代：品质优选、...
    仅「一、」「二、」下且非「替代话术」小节的词条进入严禁词；「三、」及含「合规替代」整段跳过。
    """
    words: list[dict] = []
    # None=尚未进入一二节；True=严禁区；False=已进入合规替代区，整段丢弃
    zone: Optional[bool] = None
    default_level = "high"

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # 大节标题
        if re.match(r"^一[、.,\s]", line):
            zone, default_level = True, "high"
            continue
        if re.match(r"^二[、.,\s]", line):
            zone, default_level = True, "medium"
            continue
        if re.match(r"^三[、.,\s]", line) or ("合规" in line and "替代" in line and len(line) < 48):
            zone = False
            continue
        if zone is False:
            # 合规替代话术：整段不加入严禁词
            continue

        # 小节标题行（无词条）
        if re.match(r"^[一二三四五六七八九十]+[、.,]", line) and "：" not in line:
            if zone is None:
                zone = True
            continue

        # 文档体：•xxx：词1、词2
        if "•" in line or "・" in line:
            norm = line.replace("・", "•")
            if "：" not in norm and ":" not in norm:
                continue
            sep = "：" if "：" in norm else ":"
            head, _, tail = norm.partition(sep)
            head = head.strip()
            tail = tail.strip()
            if not tail:
                continue
            # 「极限词替代」「功效词替代」等整行是合规话术，不是违禁词
            cat = head.lstrip("•").strip()
            if "替代" in cat and "违规" not in cat and "严禁" not in cat and "敏感" not in cat:
                continue
            if zone is None:
                zone = True
            for w in _split_zh_terms(tail):
                if len(w) >= 1:
                    words.append(
                        {"word": w, "level": default_level, "category": cat[:80], "note": ""}
                    )
            continue

        # 管道格式
        if "|" in line and not line.startswith("•"):
            words.extend(_parse_pipe_line(line))

    return words


def load_yjc_file() -> list[dict]:
    """
    解析 yjc.txt，支持两种格式：
    1) 管道：关键词|high|分类|备注
    2) 文档：•分类标题：词1、词2、词3（自动按顿号/逗号拆分）
    文件查找顺序：环境变量 YJC_FILE / YJC_PATH → src/yjc.txt → 仓库根 yjc.txt
    """
    global YJC_ACTIVE_PATH
    words: list[dict] = []
    chosen: Optional[Path] = None
    for candidate in _resolve_yjc_paths():
        if candidate.is_file():
            chosen = candidate
            break

    if chosen is None:
        print(
            f"--- [词库] 未找到 yjc.txt（已查找: {', '.join(str(p) for p in _resolve_yjc_paths())}），等待上传 ---"
        )
        YJC_ACTIVE_PATH = YJC_PATH
        return words

    YJC_ACTIVE_PATH = chosen
    try:
        lines = chosen.read_text(encoding="utf-8").splitlines()
        # 若存在明显「文档体」特征则走文档解析，否则走按行管道解析
        joined = "\n".join(lines)
        doc_like = "：" in joined and ("、" in joined or "，" in joined) and ("•" in joined or "・" in joined)
        if doc_like:
            words = _parse_doc_style_lines(lines)
        else:
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    words.extend(_parse_pipe_line(line))
                else:
                    # 单行无管道：整行作为一个词
                    words.append({"word": line, "level": "high", "category": "", "note": ""})

        # 去重（保留先出现的等级）
        uniq: dict[str, dict] = {}
        for item in words:
            w = item.get("word", "").strip()
            if not w:
                continue
            if w not in uniq:
                uniq[w] = item
        words = list(uniq.values())
        print(f"--- [词库] 已从 {chosen} 解析严禁词 {len(words)} 条 ---")
    except Exception as e:
        print(f"--- [错误] 词库解析失败: {e} ---")
        words = []
    return words


async def rebuild_ac_automaton():
    """
    重建 AC 自动机（从词库加载所有词及变体）。
    必须在已运行的事件循环内 await（例如 lifespan / 路由），禁止 run_until_complete。
    """
    words = load_yjc_file()
    # 清空旧自动机
    global ac_automaton
    ac_automaton = ahocorasick.Automaton()

    word_list = []
    # pyahocorasick 的 Automaton 没有 .count 属性，构建时自行统计词条数
    automaton_entry_count = 0
    for w in words:
        # 原词
        ac_automaton.add_word(w["word"], {"word": w["word"], "level": w["level"], "category": w["category"]})
        word_list.append(w["word"])
        automaton_entry_count += 1
        # 谐音变体也加入自动机
        for variant in generate_variants(w["word"]):
            if variant != w["word"]:
                ac_automaton.add_word(variant, {"word": w["word"], "level": w["level"], "category": w["category"], "variant": variant})
                automaton_entry_count += 1

    # 严禁词为 0 条时不能「零 add_word」直接 make_automaton，否则后续 iter 抛
    # Not an Aho-Corasick automaton yet / 或运行期崩溃，导致永远发不出 audit
    if automaton_entry_count == 0:
        ac_automaton.add_word(
            AC_AUTOMATON_EMPTY_SENTINEL,
            {
                "word": AC_AUTOMATON_EMPTY_SENTINEL,
                "level": "low",
                "category": "",
                "_sentinel": True,
            },
        )
        automaton_entry_count = 1
        print(
            "--- [词库] 当前无有效严禁词（请在项目根或 src 下放置 yjc.txt，或设置 YJC_FILE），"
            "已注入占位词条仅用于满足 AC 自动机构造，语义审计仍走 LLM ---"
        )

    ac_automaton.make_automaton()

    # 同时加载变体检测器
    variant_detector.load_from_words(word_list)

    # 持久化到数据库（异步，直接 await）
    await upsert_forbidden_words(words)
    print(
        f"--- [词库] AC 自动机重建完成，共 {automaton_entry_count} 个自动机词条 "
        f"（严禁词 {len(words)} 条）---"
    )


def schedule_rebuild_ac_automaton():
    """在任意线程调用：把 rebuild 投递到 FastAPI 主循环（避免 loop already running）。"""
    loop = _app_loop
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(rebuild_ac_automaton(), loop)
    else:
        # 无运行中的 app（例如单独脚本 import），退化为同步跑一遍异步协程
        asyncio.run(rebuild_ac_automaton())


class YJCWatchHandler(FileSystemEventHandler):
    """监听 yjc.txt 文件变化，自动重载"""
    def __init__(self):
        self._debounce_timer: threading.Timer | None = None
        self._debounce_lock = threading.Lock()

    def on_modified(self, event):
        if Path(event.src_path).name != "yjc.txt":
            return
        with self._debounce_lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(
                float(AppConfig.YJC_WATCH_INTERVAL),
                self._reload,
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _reload(self):
        try:
            print("--- [词库] 检测到文件变化，正在热重载... ---")
            schedule_rebuild_ac_automaton()
        except Exception as e:
            print(f"--- [错误] 词库热重载失败: {e} ---")


def start_yjc_watcher():
    """启动文件监控（仓库根与 src 目录下的 yjc.txt 任一变更即重载）"""
    handler = YJCWatchHandler()
    observer = Observer()
    scheduled = False
    for d in (REPO_ROOT, PROJECT_DIR):
        if d.is_dir():
            observer.schedule(handler, path=str(d), recursive=False)
            scheduled = True
    if scheduled:
        observer.start()
        print(f"--- [词库] 已启动 yjc.txt 监控: {REPO_ROOT}, {PROJECT_DIR} ---")
        return observer
    return None


# ================= 音频处理 =================

class AudioStreamProcessor:
    """处理 PCM 原始流：重采样、降噪、静音检测"""

    def __init__(self):
        self.buffer = np.array([], dtype=np.int16)
        self.input_rate = 16000
        self._update_filter()

    def _update_filter(self):
        self._hp_b, self._hp_a = signal.butter(2, 80 / (self.input_rate / 2), btype="highpass")

    def set_input_rate(self, sr: int):
        self.input_rate = int(sr)
        self._update_filter()

    def process(self, raw_bytes: bytes):
        in_data = np.frombuffer(raw_bytes, dtype=np.int16)
        if in_data.size > 0:
            filtered = signal.filtfilt(self._hp_b, self._hp_a, in_data.astype(np.float32))
            self.buffer = np.append(self.buffer, np.clip(filtered, -30000, 30000).astype(np.int16))

        # 最小处理长度校验 (80ms)
        if self.buffer.size < int(self.input_rate * 0.08):
            return b"", False

        current_data = self.buffer
        self.buffer = np.array([], dtype=np.int16)

        # 重采样逻辑
        if self.input_rate != AppConfig.TARGET_RATE:
            new_len = int(len(current_data) * (AppConfig.TARGET_RATE / self.input_rate))
            current_data = signal.resample(current_data, new_len).astype(np.int16)

        return current_data.tobytes(), True


# ================= 审计管理 =================

def normalize_text(text: str) -> str:
    """文本归一化：去除空格、标点、特殊符号"""
    return re.sub(r"[\s\-\_\|,，。！？!?:：;；·…]+", "", text or "")


def _strip_model_thinking(raw: str) -> str:
    """去掉常见「思考」包裹，避免干扰 JSON 解析。"""
    s = str(raw)
    bt = chr(96)  # `
    think = "think"
    redacted = "redacted_thinking"
    for pat in (
        bt + think + bt + r"[\s\S]*?" + bt + "/" + think + bt,
        "<" + think + ">" + r"[\s\S]*?</" + think + ">",
        "<" + redacted + ">" + r"[\s\S]*?</" + redacted + ">",
    ):
        s = re.sub(pat, "", s, flags=re.I)
    return s.strip()


def _balanced_json_from_open(s: str, open_idx: int) -> Optional[str]:
    """从指定位置的「{」起做括号配对（尊重 JSON 双引号字符串内的括号）。"""
    depth = 0
    n = len(s)
    i = open_idx
    in_string = False
    escape = False
    while i < n:
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx : i + 1]
        i += 1
    return None


def _json_parse_candidates(raw: str) -> list[str]:
    """枚举可能 JSON 子串：优先 ```json 围栏，再从右向左尝试每个「{」起的平衡块。"""
    s = raw.strip()
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, re.I):
        t = m.group(1).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    for idx in range(len(s) - 1, -1, -1):
        if s[idx] != "{":
            continue
        bal = _balanced_json_from_open(s, idx)
        if bal and bal not in seen:
            seen.add(bal)
            out.append(bal)
    return out


def extract_json_object(raw: str) -> Optional[dict]:
    """从模型输出中提取 JSON 对象（围栏、去思考块、多段 JSON 时优先解析末尾块）。"""
    if not raw or not str(raw).strip():
        return None
    s = _strip_model_thinking(str(raw).strip())
    if not s:
        return None
    for cand in _json_parse_candidates(s):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def fallback_loose_verdict_dict(raw: str) -> Optional[dict]:
    """
    当 extract_json_object 失败时，用正则从容错文本中提取 verdict/reason/risk，
    避免轻易退回纯词库导致语义漏判。
    """
    s = _strip_model_thinking(str(raw or ""))
    if not s.strip():
        return None
    v_m = re.search(r'"verdict"\s*:\s*"([^"]+)"', s)
    if not v_m:
        v_m = re.search(
            r'(?:verdict|结论|判定)\s*[:：]\s*["\']?\s*([合规违规违禁]{2})',
            s,
            re.I,
        )
    if not v_m:
        return None
    verdict = normalize_llm_verdict(v_m.group(1).strip())
    if verdict not in ("合规", "违规", "违禁"):
        return None
    r_m = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.DOTALL)
    reason = ""
    if r_m:
        reason = r_m.group(1).replace("\\n", " ").replace('\\"', '"').strip()
    if not reason:
        reason = "容错解析模型输出"
    rk_m = re.search(r'"risk_level"\s*:\s*"([^"]+)"', s, re.I)
    risk = rk_m.group(1).strip().lower() if rk_m else ""
    if risk not in ("high", "medium", "low", "none"):
        risk = "none" if verdict == "合规" else "medium"
    return {"verdict": verdict, "reason": reason, "risk_level": risk}


def _live_negation_window(text: str, idx: int, span: int = 36) -> bool:
    """匹配点前若明显为劝导/辟谣，则不触发高危补判。"""
    lo = max(0, idx - span)
    window = text[lo : idx + 2]
    return bool(
        re.search(
            r"(不要|切勿|请勿|拒绝|谨防|别买|别信|不可信|假的|谣言|不可能|忽悠|不支持|不经平台)",
            window,
        )
    )


def live_high_risk_fallback(text: str) -> Optional[tuple[str, str, str]]:
    """
    LLM 终审不可用时的直播场景补判：仅覆盖「极高置信度」违规/违禁形态，减少无词库时的语义漏判。
    返回 (status, risk_level, reason) 或 None。
    """
    t = (text or "").strip()
    if not t:
        return None

    rules: list[tuple[str, str, re.Pattern, str]] = [
        (
            "违禁",
            "high",
            re.compile(r"包治百病|药到病除|根治(癌症|肿瘤|乙肝|糖尿病)|疗效超过手术|胜过手术"),
            "虚假或夸大医疗功效宣传",
        ),
        (
            "违规",
            "high",
            re.compile(
                r"(赚|挣).{0,10}钱.{0,14}(刷|唰)单|"
                r"(刷|唰)单.{0,18}(头像|找我|私信|联系我|点我)|"
                r"(刷|唰)单.{0,8}(任务|兼职)"
            ),
            "诱导刷单、兼职任务或欺诈导流话术",
        ),
        (
            "违规",
            "high",
            re.compile(
                r"(私信|私聊).{0,20}(电话|手机|留电话|手机号|加微|微信号|联系方式)|"
                r"(留电话|打电话|加微信).{0,14}(私信|私聊)|"
                r"(货到付款|到付).{0,28}(私信|私聊).{0,18}(电话|手机|留)"
            ),
            "诱导私信留电话或脱离平台沟通，存在私下导流风险",
        ),
        (
            "违规",
            "high",
            re.compile(r"(内部|秘密).{0,8}(渠道|路子).{0,14}(抢|闭|私|联系)"),
            "诱导使用非公开/私下渠道或脱离平台交易",
        ),
        (
            "违规",
            "medium",
            re.compile(
                r"全网最低|史上最低|全国最低|全球最低|史低价|最低价|"
                r"仅此一家|仅此一店|错过再等|"
                r"行业(?:内)?第一|业内第一|销量第一|市场份额第一"
            ),
            "绝对化用语或误导性极限比价宣传",
        ),
        (
            "违规",
            "high",
            re.compile(
                r"(ZFB|VX|微信|支付宝).{0,22}(私下|私聊|转账).{0,14}(不走|绕过|脱离).{0,8}(平台|系统)"
            ),
            "诱导私下支付或脱离平台担保",
        ),
        (
            "违规",
            "high",
            re.compile(
                r"(支持|可以|能).{0,10}(ZFB|VX|微信|支付宝).{0,14}转账|"
                r"(ZFB|VX).{0,8}和.{0,8}(VX|ZFB).{0,12}转账"
            ),
            "宣扬私下转账或非平台担保支付",
        ),
        (
            "违禁",
            "high",
            re.compile(
                r"(三|3)\s*分钟.{0,12}(瘦|减|掉).{0,6}(斤|公斤)|"
                r"排毒.{0,8}(瘦|减|轻|掉肉)|"
                r"(瘦|减).{0,6}(十|10)\s*斤"
            ),
            "夸大减肥/排毒等功效宣传",
        ),
        (
            "违规",
            "high",
            re.compile(
                r"(精油|护肤品|化妆品|面霜|乳液|面膜|身体乳|瘦身|纤体).{0,40}"
                r"(调理|调节).{0,10}(内分泌|激素)|"
                r"(调理|调节).{0,10}(内分泌|激素).{0,40}"
                r"(精油|护肤品|化妆品|面霜|乳液|面膜|身体乳)"
            ),
            "非医疗产品宣称调理内分泌/激素等功效，存在违规功效导向",
        ),
        (
            "违规",
            "medium",
            re.compile(
                r"无效退款.{0,28}(买|送|赠|包邮)|"
                r"买.{0,8}送.{0,24}无效退款|"
                r"假一罚十.{0,12}(无效|不退)"
            ),
            "涉嫌虚假优惠或侵害消费者权益的营销组合话术",
        ),
    ]

    for status, risk, pat, desc in rules:
        m = pat.search(t)
        if m and not _live_negation_window(t, m.start()):
            return status, risk, desc
    return None


def normalize_llm_verdict(verdict: str) -> str:
    """把模型输出的 verdict 归一成 合规|违规|违禁，减少误降级到词库。"""
    v = unicodedata.normalize("NFKC", (verdict or "").strip())
    v = v.strip(' "\'「」『』')
    v = re.sub(r"[。.．]+$", "", v)
    v = v.replace("合規", "合规").replace("違規", "违规").replace("違规", "违规")
    low = v.lower()
    alias = {
        "合格": "合规",
        "通过": "合规",
        "不违规": "合规",
        "无违规": "合规",
        "合规通过": "合规",
        "正常": "合规",
        "不合规": "违规",
        "不合格": "违规",
        "确定违规": "违规",
        "严重违规": "违规",
        "禁止内容": "违禁",
    }
    if v in alias:
        return alias[v]
    if low in ("compliant", "pass", "ok", "legal"):
        return "合规"
    if low in ("violation", "violating", "non-compliant"):
        return "违规"
    if low in ("banned", "prohibited", "illicit"):
        return "违禁"
    return v


def risk_level_from_hits(hits: list[dict]) -> str:
    """从词库命中取最高风险等级。"""
    if not hits:
        return "medium"
    order = {"high": 3, "medium": 2, "low": 1, "none": 0}
    top = max(hits, key=lambda h: order.get(str(h.get("level", "low")), 0))
    lv = str(top.get("level", "medium"))
    return lv if lv in ("high", "medium", "low") else "medium"


def should_prefilter_highrisk_wordlist(text: str, hits: list[dict]) -> bool:
    """
    真实直播通用策略：仅在“高风险词命中 + 非劝导/辟谣语境”时前置直判，
    避免把明显违规句继续排队进 LLM 造成尾部超时。
    """
    if not hits:
        return False
    # 仅 high 级别触发；medium/low 仍交给 LLM 终审
    if risk_level_from_hits(hits) != "high":
        return False
    t = (text or "").strip()
    if not t:
        return False
    # 劝导/辟谣语境不做前置直判，避免误伤
    if is_compliance_advisory_tone(t):
        return False
    # 显式否定语境不做前置直判
    if re.search(r"(不要|请勿|切勿|别信|不可信|假的|谣言|不可能)", t):
        return False
    return True


def is_compliance_advisory_tone(text: str) -> bool:
    """
    LLM 不可用时的启发式：明显在「劝导合规、禁止用户踩坑」的表述，即使命中敏感词也倾向合规。
    仅作降级辅助，不能替代模型终审。
    """
    t = (text or "").strip()
    if not t:
        return False
    patterns = [
        r"不支持.{0,48}(微信|支付宝|转账|私下|私聊)",
        r"(请勿|不要|切勿).{0,20}(购买|买|信|转账|私下|点击|扫码)",
        r"不要\s*购买",
        r"大家不要",
        r"请大家.{0,12}不要",
        r"拒绝.{0,12}(私下|转账|交易)",
        r"禁止.{0,12}(私下|转账|交易)",
        r"(根据|依照).{0,12}(说明|指引).{0,8}下单",
        r"合理.{0,6}下单",
        r"谨防",
        r"警惕",
        r"提高防范",
        r"避免.{0,12}(私下|转账|受骗)",
        r"无质检.{0,8}(不要|勿)",
        r"三无.{0,8}(不要|勿|别买)",
        r"假冒.{0,8}(不要|勿)",
        r"没有产品是.{0,12}(顶尖|极致|第一)",  # 辟谣类
        r"不要相信",
        # 辟谣/否定「亏本甩卖」等话术：命中词但整体在拆穿套路
        r"这是不可能的",
        r"(不可能|不可信|假的|骗人|离谱).{0,6}$",
        r"听过.{0,16}(都说|有人).{0,56}(不可能|不可信|骗人|假的)",
        r"(亏本甩卖|跳楼价).{0,20}(不可能|不可信|骗人|假的|别信)",
        r"没什么利润.{0,12}(这是)?不可能的",
        # 劝大家别买劣质/假货、引导买正品（易与「三无」等词库命中冲突）
        r"(大家|各位|亲们).{0,12}(不要|勿|别).{0,20}(买|购买).{0,30}(三无|过期|假货|劣质)",
        r"(不要|勿|别).{0,10}买.{0,25}(三无|过期|假货|劣质)",
        r"(三无|过期|假货|劣质).{0,30}(不要|勿|别|切勿).{0,10}买",
        r"(认准|要买|请买|记得买|一定要买).{0,12}(正品|有保障|带质检|合格|大牌)",
    ]
    for pat in patterns:
        if re.search(pat, t):
            return True
    if t.count("不要") >= 2:
        return True
    return False


def is_low_information_utterance(text: str) -> bool:
    """口语停顿/填充词：对风控价值低，默认可直接合规，减少队列拥塞。"""
    t = re.sub(r"[，,。.!！?？\s]+", "", (text or "").strip())
    if not t:
        return True
    if len(t) <= 2 and t in {"嗯", "啊", "哦", "哈", "唉", "好", "行", "对", "是", "在"}:
        return True
    if len(t) <= 4 and t in {"好的", "收到", "明白", "可以", "没错", "继续", "接着", "然后"}:
        return True
    # 纯数字碎片（常见 ASR 切分噪声）
    if len(t) <= 4 and re.fullmatch(r"[0-9一二三四五六七八九十]+", t):
        return True
    return False


def split_completed_transcript(text: str) -> list[str]:
    """
    对 ASR completed 的长拼接句做轻量拆分，减少“两个语句合并后一起判定”。
    仅在存在明显编号/分句标点时拆，避免过拆导致语义损失。
    """
    raw = (text or "").strip()
    if not raw:
        return []
    # 统一分隔符
    s = raw.replace("；", "。").replace(";", "。")
    # 给中文编号前补换行：十三、 / 十四、 / 1、 / 2、
    s = re.sub(r"(?<!^)(?=(?:[一二三四五六七八九十百千]+|[0-9]{1,2})、)", "\n", s)
    parts = [p.strip(" ，,。.!！?？\t\r\n") for p in re.split(r"[。.!！?？\n]+", s)]
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return [raw]
    # 防止过拆：过短片段与后句合并
    merged: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        if len(cur) <= 5 and i + 1 < len(parts):
            merged.append((cur + "，" + parts[i + 1]).strip("，"))
            i += 2
            continue
        merged.append(cur)
        i += 1
    return merged if merged else [raw]


def is_benign_product_description(text: str) -> bool:
    """
    常见中性带货描述：材质/口感/外观/体验/普通优惠等，不应被误判成违规。
    仅作为「模型判违规后的纠偏」辅助，不单独给出最终结论。
    """
    t = (text or "").strip()
    if not t:
        return False
    # 明确风险词一票否决：出现则不走中性纠偏
    hard_risk = [
        r"包治百病|根治|药到病除|疗效超过手术|排毒|三天瘦|减.{0,4}十斤",
        r"刷单|兼职|点我|点开我的头像|私信|联系我",
        r"(私信|私聊).{0,20}(电话|手机|留电话|手机号|加微|微信号)",
        r"微信|支付宝|转账|私下|不走平台|内部渠道|秘密渠道",
        r"国家级|最高级|唯一|全网最低|史上最低|全球最低|史低价|最低价|"
        r"限时秒杀|最后\d+单|仅此一家|仅此一店|行业(?:内)?第一|业内第一|销量第一",
        r"(调理|调节).{0,10}(内分泌|激素)|内分泌",
    ]
    for p in hard_risk:
        if re.search(p, t):
            return False

    material = bool(
        re.search(
            r"(材质|面料|做工|手感|颜色|版型|尺码|口感|香味|稳定性|固色|耐用|舒适|染料|品质)",
            t,
        )
    )
    promo = bool(
        re.search(
            r"(宠粉价|折扣|优惠券|包邮|秒杀|立减|活动价|到手价|元|块|折|\d+\s*折)",
            t,
        )
        and re.search(r"(价|折|惠|元|块|包邮|秒杀|减|力度|活动)", t)
    )
    intro = bool(re.search(r"(今天|这款|这个|这件|这条|这瓶|这盒).{0,18}(推荐|分享|介绍)", t))
    service = bool(re.search(r"(满意|服务|售后|保障|正品|发货|客服)", t))
    return material or promo or intro or service


def _reason_is_echo_text(reason: str, text: str) -> bool:
    """判断理由是否在复读原句（低质量解释）。"""
    r = normalize_text(reason or "")
    t = normalize_text(text or "")
    if not r:
        return True
    if not t:
        return False
    if len(r) >= 8 and (r in t or t in r):
        return True
    if len(r) >= 10 and len(t) >= 10:
        # 粗略重叠比：理由如果大部分字符都来自原句，视为复读
        t_chars = set(t)
        overlap = sum(1 for ch in r if ch in t_chars)
        if overlap / max(1, len(r)) >= 0.9:
            return True
    return False


def _compliant_short_reason(text: str) -> str:
    """合规时的短理由：按内容分桶，避免「普通优惠」等套话误贴到服务类话术。"""
    t = (text or "").strip()
    if not t:
        return "未见明确违规导流或夸大/违禁承诺"
    if re.search(r"满意|服务|售后|保障|努力|用心|体验", t):
        return "服务/体验类表述，无违规导流或夸大承诺"
    if re.search(r"染料|固色|面料|材质|做工|品质|耐用|舒适", t):
        return "材质/工艺类中性描述，无绝对化或违禁承诺"
    if re.search(r"宠粉价|折扣|优惠券|包邮|秒杀|立减|到手价|元|块|折|价|惠", t):
        return "普通优惠类描述，无导流或虚假紧迫承诺"
    return "未见明确违规导流或夸大/违禁承诺"


def _fallback_reason(verdict: str, text: str, hits: list[dict], variant_results: list) -> str:
    """当模型理由无效时，生成可读且不复读原句的短理由。"""
    t = text or ""
    if verdict == "合规":
        return _compliant_short_reason(t)
    if verdict == "违禁":
        if re.search(r"包治百病|根治|药到病除|疗效超过手术", t):
            return "涉及虚假或夸大医疗功效宣传，属于高风险违禁"
        if re.search(r"排毒|三天瘦|减.{0,4}十斤|快速瘦", t):
            return "涉及夸大减肥/排毒功效宣传，存在明显误导风险"
        if hits:
            return f"命中高风险词“{hits[0].get('word', '')}”，语义呈现违禁导向"
        return "语句包含高风险违法/违禁导向内容"
    # 违规
    if re.search(
        r"(私信|私聊).{0,20}(电话|手机|留电话|手机号|加微|微信号)|"
        r"(留电话|打电话|加微信).{0,14}(私信|私聊)",
        t,
    ):
        return "诱导私信留电话或脱离平台沟通，属于违规导流"
    if re.search(
        r"全网最低|史上最低|全国最低|全球最低|史低价|最低价|"
        r"仅此一家|仅此一店|行业(?:内)?第一|业内第一|销量第一",
        t,
    ):
        return "绝对化用语或误导性极限比价宣传"
    if re.search(r"刷单|兼职|点我|点开我的头像|私信|联系我", t):
        return "存在刷单/导流或诱导联系话术，属于违规营销"
    if re.search(r"微信|支付宝|转账|私下交易|不走平台", t):
        return "存在私下交易或脱离平台担保导向，属于违规"
    if variant_results:
        vr = variant_results[0]
        return f"疑似变体绕过“{vr.get('original', '')}”规则，存在违规导向"
    if hits:
        return f"命中敏感词“{hits[0].get('word', '')}”且语义存在违规导向"
    return "语句含不当营销/导流导向，判定为违规"


def ac_match(text: str) -> list[dict]:
    """AC 自动机匹配，返回命中的违禁词列表"""
    hits = []
    normalized = normalize_text(text)

    for scan_text in [text, normalized]:
        try:
            for _, info in ac_automaton.iter(scan_text):
                if not isinstance(info, dict):
                    continue
                # 空词库时注入的占位词条，不参与命中
                if info.get("_sentinel") or info.get("word") == AC_AUTOMATON_EMPTY_SENTINEL:
                    continue
                hits.append(info)
        except (AttributeError, ValueError) as e:
            print(f"--- [词库] AC 匹配异常（将跳过词库命中）: {e} ---")

    # 去重
    seen = set()
    unique = []
    for h in hits:
        key = h["word"]
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def sanitize_variant_results(items: list) -> list[dict]:
    """
    清理低价值/异常变体：
    - variant/original 为空
    - 归一化后相同（如“优惠->优惠”）
    - 重复项去重
    """
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        v = str(it.get("variant", "") or "").strip()
        o = str(it.get("original", "") or "").strip()
        m = str(it.get("method", "") or "").strip()
        if not v or not o:
            continue
        if normalize_text(v) == normalize_text(o):
            continue
        k = (normalize_text(v), normalize_text(o), m)
        if k in seen:
            continue
        seen.add(k)
        out.append({"variant": v, "original": o, "method": m})
    return out


class LiveAuditManager:
    """
    单个 WebSocket 连接的审计管理器
    每个连接有独立的 ASR 实例、心跳、审计队列
    """

    def __init__(self, websocket: WebSocket, room_id: str = "", client_ip: str = ""):
        self.session_id = str(uuid.uuid4())[:8]
        self.ws = websocket
        self.room_id = room_id or f"room_{self.session_id}"
        self.client_ip = client_ip
        # DashScope ASR 回调在独立线程执行，必须用「创建 Manager 时」的 FastAPI 事件循环做 thread-safe 投递
        self.loop = asyncio.get_running_loop()
        _to = httpx.Timeout(
            connect=30.0,
            read=AppConfig.OLLAMA_HTTP_READ_TIMEOUT,
            write=120.0,
            pool=30.0,
        )
        self.http_client = httpx.AsyncClient(timeout=_to)
        self.is_active = True
        self.last_activity = time.time()
        self._audio_ended = False
        self.conv = None
        # 上下文记忆（默认仅保留少量历史，兼顾准确率与时延）
        self.context_history = []
        self.MAX_CONTEXT = AppConfig.LLM_CONTEXT_TURNS
        # 同句短期缓存：直播口播高频重复，直接复用判定可显著提速并保持一致性
        self._verdict_cache: dict[str, tuple[float, str, str, str, str]] = {}
        # 直播场景复读较多：默认适度延长缓存可明显减少重复 LLM 请求
        self._verdict_cache_ttl = float(os.getenv("LLM_VERDICT_CACHE_TTL_SEC", "300"))
        # 并发同句去重：同一时刻同句只发一次 LLM 请求，其他协程复用结果（不影响准确率）
        self._llm_inflight: dict[str, asyncio.Task] = {}
        # 跟踪审计执行状态：高/低优先级双队列，保障高风险句先判
        pri_cap = max(1, int(AppConfig.MAX_QUEUE_SIZE * AppConfig.AUDIT_PRIORITY_QUEUE_RATIO))
        norm_cap = max(1, AppConfig.MAX_QUEUE_SIZE - pri_cap)
        self._audit_queue_hi: asyncio.Queue[tuple[str, str, float]] = asyncio.Queue(
            maxsize=pri_cap
        )
        self._audit_queue_norm: asyncio.Queue[tuple[str, str, float]] = asyncio.Queue(
            maxsize=norm_cap
        )
        self._audit_workers: list[asyncio.Task] = []
        self._audit_inflight = 0
        # 最近若干句用于跨句组合违规检测（真实直播常跨句导流）
        self._recent_utterances: deque[str] = deque(maxlen=8)
        # ASR 终稿分段 id（在回调线程分配，须加锁）
        self._segment_lock = threading.Lock()
        self._segment_seq = 0
        self._asr_last_reconnect_ts = 0.0
        self._asr_keepalive_task: Optional[asyncio.Task] = None
        self._asr_session_updated = threading.Event()
        self._asr_wire_log_until: float = 0.0
        self._asr_closed_before_ready: bool = False

        # 持久化：记录会话开始
        asyncio.create_task(self._init_persistence())

        # 启动心跳检查
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # 启动音频消费者（ASR 回调推送的音频数据）
        self._init_qwen_asr()
        if AppConfig.ASR_KEEPALIVE_ENABLED:
            self._asr_keepalive_task = asyncio.create_task(self._asr_keepalive_loop())
        # 启动审计 worker（从队列顺序消费，减少 LLM 任务洪峰）
        for i in range(AppConfig.AUDIT_WORKERS_PER_SESSION):
            wt = asyncio.create_task(self._audit_worker_loop(i))
            self._audit_workers.append(wt)

        print(f"--- [会话] {self.session_id} 已创建, 房间: {self.room_id} ---")

    def _audit_backlog(self) -> int:
        return self._audit_queue_hi.qsize() + self._audit_queue_norm.qsize()

    def mark_audio_ended(self) -> None:
        """客户端已发 end：后续以“尽快出清在途审计”为优先。"""
        self._audio_ended = True

    def _is_high_priority_text(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return bool(
            re.search(
                r"刷单|兼职|私信|私聊|留电话|加微|微信|转账|不走平台|"
                r"包治百病|根治|排毒|三天瘦|内分泌|全网最低|仅此一家|行业(?:内)?第一",
                t,
            )
        )

    def _cross_sentence_violation(self, text: str) -> Optional[tuple[str, str, str]]:
        """
        跨句组合违规：单句看似普通，合并上下文后才构成导流/违规承诺。
        """
        t = (text or "").strip()
        if not t:
            return None
        # 只看「前两句」，且必须由“当前句”贡献一半证据，避免连坐误判
        recent = list(self._recent_utterances)
        if recent and recent[-1] == t:
            recent = recent[:-1]
        prev = " ".join(recent[-2:])
        if not prev:
            return None

        cur_channel = bool(re.search(r"私信|私聊|脱离平台|不走平台|内部渠道|秘密渠道", t))
        cur_contact = bool(re.search(r"电话|手机|留电话|手机号|加微|微信号|联系方式|加微信", t))
        prev_channel = bool(re.search(r"私信|私聊|脱离平台|不走平台|内部渠道|秘密渠道", prev))
        prev_contact = bool(re.search(r"电话|手机|留电话|手机号|加微|微信号|联系方式|加微信", prev))

        # 组合 1：当前句与前句拼出「私下沟通 + 联系方式」
        if (cur_channel and prev_contact) or (cur_contact and prev_channel):
            return ("违规", "high", "跨句出现私信沟通与联系方式，存在私下导流风险")

        cur_effect = bool(re.search(r"调理内分泌|提高免疫力|排毒|快速瘦|治疗|根治|疗效", t))
        cur_product = bool(re.search(r"精油|护肤品|化妆品|面膜|身体乳|理疗|裤|食品|保健", t))
        prev_effect = bool(re.search(r"调理内分泌|提高免疫力|排毒|快速瘦|治疗|根治|疗效", prev))
        prev_product = bool(re.search(r"精油|护肤品|化妆品|面膜|身体乳|理疗|裤|食品|保健", prev))

        # 组合 2：当前句与前句拼出「商品 + 功效承诺」
        if (cur_effect and prev_product) or (cur_product and prev_effect):
            return ("违规", "high", "跨句组合形成功效承诺，存在违规宣传风险")
        return None

    def _quick_preview_verdict(
        self,
        text: str,
        hits: list[dict],
        variant_results: list,
    ) -> tuple[str, str, str]:
        """
        快判（仅用于提前反馈，不作为最终结论）：
        - 命中高风险规则：直接违规
        - 命中敏感词/变体：倾向违规待复核
        - 其余：合规待复核
        """
        lur = live_high_risk_fallback(text)
        if lur:
            st, rk, desc = lur
            return st, rk, f"[快判] {desc}（待终审复核）"
        if hits or variant_results:
            return "违规", risk_level_from_hits(hits), "[快判] 命中风险线索（待终审复核）"
        return "合规", "none", "[快判] 未见显著风险线索（待终审复核）"

    def _estimate_verdict_confidence(
        self,
        verdict: Optional[str],
        text: str,
        reason: str,
        hits: list[dict],
        variant_results: list,
    ) -> float:
        """轻量置信度估计：仅用于决定是否触发终审复核。"""
        if verdict not in ("合规", "违规", "违禁"):
            return 0.0
        score = 0.55
        lur = live_high_risk_fallback(text)
        if verdict in ("违规", "违禁") and (hits or variant_results or lur):
            score += 0.25
        if verdict == "合规" and (not hits) and (not variant_results) and is_benign_product_description(text):
            score += 0.20
        if _reason_is_echo_text(reason, text):
            score -= 0.20
        if len((reason or "").strip()) < 6:
            score -= 0.10
        return max(0.0, min(1.0, score))

    def _need_recheck(
        self,
        verdict: Optional[str],
        text: str,
        reason: str,
        hits: list[dict],
        variant_results: list,
        *,
        backlog: int = 0,
        queue_wait_ms: float = 0.0,
    ) -> bool:
        """快判冲突/低置信时触发终审复核。"""
        if verdict not in ("合规", "违规", "违禁"):
            return True
        lur = live_high_risk_fallback(text)
        # 劝导/辟谣语境 + 快判合规：不再强制拉终审大模型，显著降低排队与超时
        if verdict == "合规" and hits and is_compliance_advisory_tone(text) and lur is None:
            return False
        if lur and verdict == "合规":
            return True
        if (
            risk_level_from_hits(hits) == "high"
            and verdict == "合规"
            and not is_compliance_advisory_tone(text)
        ):
            return True
        if verdict in ("违规", "违禁") and (not hits) and (not variant_results) and is_benign_product_description(text):
            return True
        dyn_thr = AppConfig.FAST_RECHECK_CONFIDENCE
        overloaded = (
            backlog >= AppConfig.RECHECK_GUARD_BACKLOG
            or queue_wait_ms >= AppConfig.RECHECK_GUARD_QUEUE_MS
        )
        if AppConfig.ENABLE_DYNAMIC_RECHECK_GUARD and overloaded:
            dyn_thr += AppConfig.RECHECK_GUARD_BOOST
            if (
                backlog >= (AppConfig.RECHECK_GUARD_BACKLOG * 2)
                or queue_wait_ms >= (AppConfig.RECHECK_GUARD_QUEUE_MS * 2)
            ):
                dyn_thr += AppConfig.RECHECK_GUARD_BOOST_HEAVY
            dyn_thr = min(0.95, dyn_thr)
            # 拥塞保护：合规快判 + 非高危规则命中时，优先放行避免尾部雪崩
            if verdict == "合规" and lur is None:
                return False
        return self._estimate_verdict_confidence(verdict, text, reason, hits, variant_results) < dyn_thr

    async def _audit_worker_loop(self, worker_idx: int):
        """顺序消费审计队列，避免 dispatch 阶段创建海量并发任务。"""
        hi_burst = 0
        while self.is_active or (self._audit_backlog() > 0):
            used_q: Optional[asyncio.Queue] = None
            enqueued_at = 0.0
            try:
                prefer_hi = hi_burst < AppConfig.AUDIT_HI_BURST
                if prefer_hi:
                    try:
                        text, segment_id, enqueued_at = self._audit_queue_hi.get_nowait()
                        used_q = self._audit_queue_hi
                    except asyncio.QueueEmpty:
                        text, segment_id, enqueued_at = self._audit_queue_norm.get_nowait()
                        used_q = self._audit_queue_norm
                else:
                    try:
                        text, segment_id, enqueued_at = self._audit_queue_norm.get_nowait()
                        used_q = self._audit_queue_norm
                    except asyncio.QueueEmpty:
                        text, segment_id, enqueued_at = self._audit_queue_hi.get_nowait()
                        used_q = self._audit_queue_hi
            except asyncio.QueueEmpty:
                # 两队列均暂空：按公平策略等待，避免仅盯高优导致普通队列尾延迟放大
                try:
                    if hi_burst >= AppConfig.AUDIT_HI_BURST:
                        text, segment_id, enqueued_at = await asyncio.wait_for(
                            self._audit_queue_norm.get(), timeout=1.0
                        )
                        used_q = self._audit_queue_norm
                    else:
                        text, segment_id, enqueued_at = await asyncio.wait_for(
                            self._audit_queue_hi.get(), timeout=1.0
                        )
                        used_q = self._audit_queue_hi
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except asyncio.QueueEmpty:
                    text, segment_id, enqueued_at = await asyncio.wait_for(
                        self._audit_queue_norm.get(), timeout=1.0
                    )
                    used_q = self._audit_queue_norm
            except asyncio.CancelledError:
                break
            try:
                if used_q is self._audit_queue_hi:
                    hi_burst += 1
                else:
                    hi_burst = 0
                self._audit_inflight += 1
                wait_ms = max(0.0, (time.time() - enqueued_at) * 1000.0)
                await self._execute_audit(text, segment_id, queue_wait_ms=wait_ms)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"--- [审计异常] {self.session_id}/w{worker_idx}: {e} ---")
            finally:
                self._audit_inflight = max(0, self._audit_inflight - 1)
                if used_q is not None:
                    used_q.task_done()

    async def _ollama_post(self, url: str, json_body: dict) -> httpx.Response:
        """Ollama HTTP：对超时/断连等瞬态错误退避重试，提高 LLM 可用率。"""
        attempts = 1 + AppConfig.OLLAMA_TRANSIENT_RETRIES
        last_exc: Optional[BaseException] = None
        for i in range(attempts):
            try:
                return await self.http_client.post(url, json=json_body)
            except (
                httpx.ReadTimeout,
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ) as e:
                last_exc = e
                if i + 1 >= attempts:
                    raise
                await asyncio.sleep(min(2.0, 0.25 * (2**i)))
        raise last_exc  # pragma: no cover

    def allocate_segment_id(self) -> str:
        """分配本条 ASR 终稿的 segment_id（线程安全）。格式：{session_id}-t{序号6位}。"""
        with self._segment_lock:
            self._segment_seq += 1
            n = self._segment_seq
        return f"{self.session_id}-t{n:06d}"

    async def _emit_transcript_final(self, segment_id: str, text: str) -> None:
        """终稿先于审计推送，便于客户端用 segment_id 绑定 UI 行。"""
        if not self.is_active:
            return
        try:
            await self.ws.send_json(
                {
                    "type": "transcript_final",
                    "segment_id": segment_id,
                    "text": text,
                    "room_id": self.room_id,
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            pass

    async def _init_persistence(self):
        await save_session(self.session_id, self.room_id, self.client_ip)

    def _init_qwen_asr(self):
        """初始化 Qwen-Omni 实时流（模型名来自 AppConfig / 环境变量，可选备用名）"""
        if self.conv is not None:
            try:
                self.conv.close()
            except Exception:
                pass
            self.conv = None

        outer = self

        class AuditCallback(OmniRealtimeCallback):
            def on_close(self, close_status_code, close_msg):
                if not outer._asr_session_updated.is_set():
                    outer._asr_closed_before_ready = True
                try:
                    if isinstance(close_msg, (bytes, bytearray)):
                        cmsg = close_msg.decode("utf-8", errors="replace")
                    else:
                        cmsg = str(close_msg or "")
                except Exception:
                    cmsg = repr(close_msg)
                print(
                    f"--- [ASR upstream on_close] {outer.session_id} "
                    f"code={close_status_code} msg={cmsg[:500]} ---"
                )
                if not outer._asr_session_updated.is_set():
                    print(
                        f"--- [ASR 提示] {outer.session_id} 连接在收到 session.updated 前已关闭。"
                        "请核对：国际控制台 Key → apikey.env 设置 "
                        "DASHSCOPE_REALTIME_WS_URL=wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime ；"
                        "北京 Key → 删除该变量或设为 "
                        "wss://dashscope.aliyuncs.com/api-ws/v1/realtime 。"
                        "并确认 URL 勿带 ?model=（由 SDK 自动拼接）。---"
                    )

            def on_event(self, response):
                et = response.get("type")
                if time.time() < outer._asr_wire_log_until:
                    try:
                        if et == "error":
                            print(
                                f"--- [ASR wire] {outer.session_id} type=error "
                                f"detail={json.dumps(response.get('error'), ensure_ascii=False)[:500]} ---"
                            )
                        else:
                            print(f"--- [ASR wire] {outer.session_id} type={et!r} ---")
                    except Exception:
                        pass
                if et == "session.updated":
                    outer._asr_session_updated.set()
                elif et == "error":
                    try:
                        detail = json.dumps(response, ensure_ascii=False)[:2000]
                    except Exception:
                        detail = str(response)[:2000]
                    print(f"--- [ASR upstream error] {outer.session_id} {detail} ---")
                # 1. 终稿转写：必须在 is_active 判断之前投递，否则 end→close 过快时会永远收不到 audit
                if response.get("type") == "conversation.item.input_audio_transcription.completed":
                    final_text = response.get("transcript", "").strip()
                    if final_text:
                        outer.last_activity = time.time()
                        chunks = split_completed_transcript(final_text)
                        for i, chunk in enumerate(chunks):
                            seg = outer.allocate_segment_id()
                            if i > 0:
                                print(
                                    f"--- [ASR split] {outer.session_id} "
                                    f"orig={final_text[:80]!r} -> chunk={chunk[:80]!r} ---"
                                )
                            asyncio.run_coroutine_threadsafe(
                                outer.dispatch_audit(chunk, seg),
                                outer.loop,
                            )

                if not outer.is_active:
                    return

                # 2. 推送实时中间结果
                if "transcript" in response:
                    partial = response.get("transcript", "").strip()
                    if partial:
                        outer.last_activity = time.time()
                        asyncio.run_coroutine_threadsafe(
                            outer.ws.send_json({"type": "partial", "text": partial}),
                            outer.loop,
                        )

        models = [AppConfig.DASHSCOPE_ASR_MODEL]
        fb = AppConfig.DASHSCOPE_ASR_MODEL_FALLBACK
        if fb and fb not in models:
            models.append(fb)

        api_key_use = (os.getenv("DASHSCOPE_API_KEY") or dashscope.api_key or "").strip() or None

        last_exc: Optional[Exception] = None
        for i, model_name in enumerate(models):
            try:
                self._asr_closed_before_ready = False
                conv_kw: dict = {
                    "model": model_name,
                    "callback": AuditCallback(),
                    "api_key": api_key_use,
                }
                ws_base = AppConfig.DASHSCOPE_REALTIME_WS_URL
                if ws_base:
                    ws_base = ws_base.strip()
                    if "?" in ws_base:
                        ws_base = ws_base.split("?", 1)[0].rstrip("/")
                        print(
                            "--- [DashScope] DASHSCOPE_REALTIME_WS_URL 不应含 ?query；"
                            "已去掉查询串，由 SDK 自动附加 model= ---"
                        )
                    conv_kw["url"] = ws_base
                if AppConfig.DASHSCOPE_WORKSPACE_ID:
                    conv_kw["workspace"] = AppConfig.DASHSCOPE_WORKSPACE_ID
                self.conv = OmniRealtimeConversation(**conv_kw)
                self.conv.connect()
                self._asr_session_updated.clear()
                self._asr_wire_log_until = time.time() + min(
                    30.0, max(8.0, float(AppConfig.ASR_SESSION_UPDATED_TIMEOUT_SEC) + 5.0)
                )
                # 与官方示例一致，勿额外传 input_audio_transcription：部分地域网关对带 model 的覆盖会拒会话，导致无 session.updated
                _sess_upd = dict(
                    output_modalities=[MultiModality.TEXT],
                    enable_input_audio_transcription=True,
                    transcription_params=TranscriptionParams(
                        language="zh",
                        sample_rate=16000,
                        input_audio_format="pcm",
                    ),
                )
                if not AppConfig.ASR_SERVER_VAD:
                    _sess_upd["enable_turn_detection"] = False
                self.conv.update_session(**_sess_upd)
                t_ready = max(2.0, float(AppConfig.ASR_SESSION_UPDATED_TIMEOUT_SEC))
                if not self._asr_session_updated.wait(timeout=t_ready):
                    extra = ""
                    if self._asr_closed_before_ready:
                        extra = (
                            " 上游已在就绪前断开，请优先核对 WebSocket 区域与 Key 区域是否一致（见 on_close 上方提示）。"
                        )
                    elif not AppConfig.DASHSCOPE_REALTIME_WS_URL:
                        extra = (
                            " 当前使用 SDK 默认北京 WebSocket；若 Key 在国际控制台申请，"
                            "请在 apikey.env 设置 DASHSCOPE_REALTIME_WS_URL="
                            "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime 。"
                        )
                    raise TimeoutError(
                        f"DashScope 在 {t_ready:.0f}s 内未返回 session.updated，禁止提前 append_audio。"
                        f"{extra}"
                        " 另请核对 DASHSCOPE_API_KEY、DASHSCOPE_ASR_MODEL、本机出网与 dashscope SDK 版本；"
                        "或增大 ASR_SESSION_UPDATED_TIMEOUT_SEC。"
                    )
                if i > 0:
                    print(
                        f"--- [ASR] {self.session_id} 主模型不可用，已用备用 DASHSCOPE_ASR_MODEL_FALLBACK={model_name} ---"
                    )
                return
            except Exception as e:
                last_exc = e
                if self.conv is not None:
                    try:
                        self.conv.close()
                    except Exception:
                        pass
                    self.conv = None
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("ASR 初始化未尝试任何模型")

    async def reconnect_qwen_asr(self, reason: str = "") -> bool:
        """DashScope 上游断链后重连，成功返回 True。"""
        if not self.is_active:
            return False
        now = time.time()
        cooldown = max(
            0.0,
            float(getattr(AppConfig, "ASR_UPSTREAM_RECONNECT_COOLDOWN_SEC", 1.0)),
        )
        if now - self._asr_last_reconnect_ts < cooldown:
            return False
        self._asr_last_reconnect_ts = now
        retries = max(0, int(getattr(AppConfig, "ASR_UPSTREAM_RECONNECT_RETRIES", 2)))
        backoff = float(getattr(AppConfig, "ASR_UPSTREAM_RECONNECT_BACKOFF_SEC", 1.0))
        for i in range(retries + 1):
            try:
                self._init_qwen_asr()
                return True
            except Exception as e:
                if i >= retries:
                    print(
                        f"--- [ASR 重连失败] {self.session_id} reason={reason} "
                        f"attempt={i+1}/{retries+1} err={type(e).__name__}: {e} ---"
                    )
                    return False
                await asyncio.sleep(min(3.0, backoff * (2**i)))
        return False

    async def _llm_semantic_verdict(
        self,
        text: str,
        hits: list[dict],
        variant_results: list,
        *,
        fast_mode: bool = False,
        model_override: Optional[str] = None,
    ) -> tuple[Optional[str], str, str, str, float, float]:
        """
        全进程限流调用 Ollama（两段预算）：
        1) sem_wait：等待全局并发槽位预算
        2) infer：拿到槽位后的推理预算
        返回附带 sem_wait_ms / infer_ms
        """
        sem_t0 = time.time()
        acquired = False
        try:
            await asyncio.wait_for(
                _global_ollama_sem.acquire(),
                timeout=max(0.1, AppConfig.LLM_SEM_WAIT_TIMEOUT_SEC),
            )
            acquired = True
            sem_wait_ms = (time.time() - sem_t0) * 1000.0
        except asyncio.TimeoutError:
            sem_wait_ms = (time.time() - sem_t0) * 1000.0
            return (
                None,
                f"[LLM] 并发槽位等待超时（>{AppConfig.LLM_SEM_WAIT_TIMEOUT_SEC:.0f}s）",
                "none",
                "",
                sem_wait_ms,
                0.0,
            )

        infer_t0 = time.time()
        try:
            timeout_sec = (
                AppConfig.LLM_FAST_TIMEOUT_SEC if fast_mode else AppConfig.LLM_SINGLE_TIMEOUT_SEC
            )
            if fast_mode and not AppConfig.LLM_USE_FAST_MODEL:
                # 快慢共用 OLLAMA_MODEL 时，快判与终审算力相同，10s 级预算会导致长句必触发假超时
                timeout_sec = max(
                    timeout_sec,
                    min(float(AppConfig.LLM_SINGLE_TIMEOUT_SEC), 90.0),
                )

            # 使用指定模型或默认模型
            model_to_use = model_override if model_override else llm_chat_model_name(fast_mode)

            verdict, reason, risk, raw = await asyncio.wait_for(
                self._llm_semantic_verdict_core(
                    text, hits, variant_results, fast_mode=fast_mode, model_override=model_to_use
                ),
                timeout=timeout_sec,
            )
            infer_ms = (time.time() - infer_t0) * 1000.0
            return verdict, reason, risk, raw, sem_wait_ms, infer_ms
        except asyncio.TimeoutError:
            infer_ms = (time.time() - infer_t0) * 1000.0
            if (not fast_mode) and AppConfig.LLM_TIMEOUT_RETRY_ONCE:
                retry_timeout = min(
                    120.0,
                    max(timeout_sec + float(AppConfig.LLM_TIMEOUT_RETRY_EXTRA_SEC), timeout_sec + 6.0),
                )
                retry_t0 = time.time()
                try:
                    verdict, reason, risk, raw = await asyncio.wait_for(
                        self._llm_semantic_verdict_core(
                            text,
                            hits,
                            variant_results,
                            fast_mode=False,
                            model_override=model_override,
                            force_compact=True,
                        ),
                        timeout=retry_timeout,
                    )
                    infer_ms += (time.time() - retry_t0) * 1000.0
                    return verdict, f"{reason} [retry_compact]", risk, raw, sem_wait_ms, infer_ms
                except asyncio.TimeoutError:
                    infer_ms += (time.time() - retry_t0) * 1000.0
                except Exception:
                    infer_ms += (time.time() - retry_t0) * 1000.0
            return (
                None,
                f"[LLM] 单句终审超时（>{timeout_sec:.0f}s）",
                "none",
                "",
                sem_wait_ms,
                infer_ms,
            )
        finally:
            if acquired:
                _global_ollama_sem.release()

    async def _ollama_minimal_verdict_json(
        self,
        text: str,
        *,
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """
        主请求 JSON 解析失败时的极简二次请求：短上下文、低温、短输出；同样尝试 JSON Schema。
        """
        compact = (text or "").strip()[:650]
        m = (model or AppConfig.OLLAMA_MODEL).strip()
        nc = int(num_ctx or AppConfig.LLM_AUDIT_NUM_CTX)
        npred = int(num_predict or min(200, AppConfig.OLLAMA_NUM_PREDICT))
        sp = (
            "直播合规终审。只输出一个 JSON 对象，第一个字符必须是{。"
            "键：verdict、reason、risk_level。"
            "verdict 只能是中文：合规、违规、违禁 三选一。"
            "risk_level 只能是英文小写：none、low、medium、high 之一。"
            "禁止 Markdown、禁止代码围栏、禁止思考过程。"
        )
        base = {
            **ollama_request_top_level_extras(),
            "model": m,
            "messages": [
                {"role": "system", "content": sp},
                {"role": "user", "content": "主播原话：\n" + compact},
            ],
            "stream": False,
            "options": {
                "num_ctx": nc,
                "temperature": 0.01,
                "num_predict": npred,
                "top_p": AppConfig.OLLAMA_TOP_P,
                "repeat_penalty": AppConfig.OLLAMA_REPEAT_PENALTY,
            },
        }
        # 极简补判：强制关思考，专出短 JSON，避免与主请求 think 设置叠加导致难解析
        base["think"] = False
        try:
            for fmt in ollama_verdict_format_sequence():
                p = dict(base)
                if fmt is None:
                    p.pop("format", None)
                else:
                    p["format"] = fmt
                r = await self._ollama_post(AppConfig.OLLAMA_CHAT_URL, p)
                if r.status_code != 200:
                    if r.status_code not in (400, 422):
                        return ""
                    continue
                msg = (r.json() or {}).get("message") or {}
                return merge_ollama_chat_message_parts(msg)
            return ""
        except Exception:
            return ""

    async def _llm_semantic_verdict_core(
        self,
        text: str,
        hits: list[dict],
        variant_results: list,
        *,
        fast_mode: bool = False,
        model_override: Optional[str] = None,
        force_compact: bool = False,
    ) -> tuple[Optional[str], str, str, str]:
        """
        语言模型终审：命中词/变体仅作参考，必须结合语境。
        返回 (verdict, reason, risk_level, raw_content)；verdict ∈ {合规,违规,违禁}；解析失败 verdict=None。
        """
        hit_words = list(dict.fromkeys([h["word"] for h in hits if h.get("word")]))
        variant_lines: list[str] = []
        for vr in variant_results or []:
            if isinstance(vr, dict):
                variant_lines.append(
                    f"变体「{vr.get('variant', '')}」疑似对应原词「{vr.get('original', '')}」"
                    f"（{vr.get('method', '')}）"
                )

        ref = ""
        if hit_words:
            ref += "【词库可能命中（仅参考，命中≠违规）】" + "、".join(hit_words[:32]) + "\n"
        if variant_lines:
            ref += "【变体检测线索】\n" + "\n".join(variant_lines[:8]) + "\n"

        model_name = model_override if model_override else llm_chat_model_name(fast_mode)

        use_fast_prompt = fast_mode or AppConfig.LLM_FAST_PROMPT
        if use_fast_prompt:
            sys_msg = (
                "你是直播内容合规审核员。仅依据主播原话判定，勿臆测。\n"
                "只输出一个JSON：{\"verdict\":\"合规|违规|违禁\",\"reason\":\"...\",\"risk_level\":\"none|low|medium|high\"}\n"
                "规则：劝导/辟谣/警示=合规；私下交易导流、虚假紧迫感、刷单兼职=违规。\n"
                "【绝对化/极限用语（营销口播）】主播在推销商品/服务/价格/效果时，出现明显排他或顶格承诺，"
                "默认判违规（verdict=违规），除非整句明确是辟谣/劝导/客观科普且无推销意图。\n"
                "典型触发（出现其一且语境为带货/促销即高危）：全网最低/全国最低/史低价/底价/仅此一天/错过再等一年、"
                "唯一/独家/首家/第一/NO.1/冠军/顶级/极致/最强/最好/最优/百分百有效/一定有效/保证治愈、"
                "永久/终身/从不反弹/无效退款（夸大疗效承诺）、全球首发/独一无二（用于夸大稀缺排他）等。\n"
                "虚假医疗功效（包治百病/根治重疾/痊愈水肿等治疗承诺）与违法高危内容=违禁。"
                "命中敏感词不等于违规，必须看语境；reason<=36字，描述违规点，不可照抄整句原话。"
                "若仅是材质/口感/外观/普通优惠描述且无导流承诺，优先判合规。"
                "非医疗产品若宣称调理/调节内分泌、激素等身体功能，判违规。"
                "引导私信/私聊留电话、加微信、脱离平台沟通，判违规。"
            )
        else:
            sys_msg = (
                "你是中国大陆直播平台资深内容合规审核专家，拥有多年直播风控实战经验。\n"
                "任务：仅依据「主播原话」与「参考命中」做终审判定，严禁臆测未说出的内容。\n\n"
                "【核心原则】准确区分「口播出现敏感词」与「主播诱导违规营销」。\n"
                "• 出现敏感词 ≠ 违规：关键看语境是否为劝导、辟谣、警示、中性说明。\n"
                "• 诱导违规 = 违规：关键看是否有明确的违规营销意图或违法暗示。\n\n"
                "【绝对化/极限用语（重点）】凡在带货、促销、功效承诺语境下，使用排他性、顶格级、无条件承诺，"
                "足以误导消费者的表述，判违规；不要因后半句出现中性描述而整体放行。\n"
                "常见形态：最/第一/唯一/独家/首家/全网/全国/史低/底价/仅此/错过再等/百分百/一定/保证、"
                "全球首发/独一无二（夸大稀缺排他）、永久/终身/从不反弹等。\n"
                "例外（可合规）：整句明确为平台规则宣读、辟谣「并非唯一/并非最低」、纯科普无推销且无购买引导。\n\n"
                "【输出格式】只输出一个JSON对象，首字符{末字符}，禁止Markdown/代码围栏。\n"
                "键：verdict∈{合规,违规,违禁}，reason（<=36字，说明违规点，不要复读整句），risk_level∈{none,low,medium,high}。\n\n"
                "【判定标准详解】\n"
                "合规：劝导、辟谣、中性说明。\n"
                "违规：绝对化与极限宣传、私下交易导流、逼单、刷单、不当背书。\n"
                "违禁：虚假医疗功效（含痊愈/治愈/根治/消除病症类承诺）、违法高危内容。\n"
                "确认无误后只输出JSON。\n"
            )

        # 构建包含上下文的用户消息（提升准确率的关键）；快判不带历史以降低 prefill
        if fast_mode:
            history_turns: list[str] = []
        else:
            history_turns = self.context_history[:-1]
            if len(history_turns) > max(0, self.MAX_CONTEXT - 1):
                history_turns = history_turns[-max(0, self.MAX_CONTEXT - 1) :]
        n_ctx, n_predict = llm_num_ctx_predict(
            fast_mode,
            text_len=len((text or "").strip()),
            history_turns=len(history_turns),
            force_compact=force_compact,
        )
        context_str = "".join([f"[历史{i+1}] {ctx}\n" for i, ctx in enumerate(history_turns)])
        if context_str:
            context_str = "【上下文历史】\n" + context_str

        text_cap = 700 if fast_mode else 1200
        user_msg = (
            context_str +
            ref
            + "【本轮待审原话】\n"
            + (text.strip()[:text_cap])
        )
        base_payload = {
            **llm_chat_top_level_extras(fast_mode),
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {
                "num_ctx": n_ctx,
                "temperature": 0.02,
                "num_predict": n_predict,
                "top_p": AppConfig.OLLAMA_TOP_P,
                "repeat_penalty": AppConfig.OLLAMA_REPEAT_PENALTY,
            },
        }
        raw_content = ""
        resp = None
        last_chat_message: Optional[dict] = None

        try:
            chat_ok = False
            for fmt in ollama_verdict_format_sequence():
                p = dict(base_payload)
                if fmt is None:
                    p.pop("format", None)
                else:
                    p["format"] = fmt
                resp = await self._ollama_post(AppConfig.OLLAMA_CHAT_URL, p)
                if resp.status_code == 200:
                    msg = (resp.json() or {}).get("message") or {}
                    last_chat_message = msg if isinstance(msg, dict) else None
                    raw_content = merge_ollama_chat_message_parts(msg)
                    chat_ok = True
                    break
                if resp.status_code not in (400, 422):
                    break

            if not chat_ok and resp is not None and resp.status_code == 404:
                gen_base = {
                    **llm_chat_top_level_extras(fast_mode),
                    "model": model_name,
                    "prompt": sys_msg
                    + "\n\n"
                    + user_msg
                    + '\n\n只输出JSON: {"verdict":"...","reason":"...","risk_level":"..."}',
                    "stream": False,
                    "options": {
                        "num_ctx": n_ctx,
                        "temperature": 0.02,
                        "num_predict": n_predict,
                        "top_p": AppConfig.OLLAMA_TOP_P,
                        "repeat_penalty": AppConfig.OLLAMA_REPEAT_PENALTY,
                    },
                }
                for fmt in ollama_verdict_format_sequence():
                    gp = dict(gen_base)
                    if fmt is None:
                        gp.pop("format", None)
                    else:
                        gp["format"] = fmt
                    gen_resp = await self._ollama_post(
                        AppConfig.OLLAMA_GENERATE_URL, gp
                    )
                    if gen_resp.status_code == 200:
                        raw_content = (gen_resp.json() or {}).get("response", "") or ""
                        resp = gen_resp
                        break
                    if gen_resp.status_code not in (400, 422):
                        break
            elif not chat_ok and resp is not None and resp.status_code >= 400:
                print(
                    f"--- [LLM] HTTP {resp.status_code} chat 失败: "
                    f"{(resp.text or '')[:400]} ---"
                )
        except Exception as e:
            print(f"--- [LLM] 请求异常详情: {type(e).__name__}: {e} ---")
            return None, f"[LLM] 请求异常: {type(e).__name__}: {e}", "none", raw_content

        obj = None
        if last_chat_message is not None:
            for cand in iter_verdict_parse_strings_from_chat(last_chat_message):
                obj = extract_json_object(cand) or fallback_loose_verdict_dict(cand)
                if obj:
                    break
        if not obj:
            obj = extract_json_object(raw_content) or fallback_loose_verdict_dict(
                raw_content
            )
        if not obj and AppConfig.LLM_PARSE_RETRY:
            retry_raw = await self._ollama_minimal_verdict_json(
                text,
                model=model_name,
                num_ctx=n_ctx,
                num_predict=min(200, n_predict),
            )
            if retry_raw.strip():
                obj = extract_json_object(retry_raw) or fallback_loose_verdict_dict(
                    retry_raw
                )
                if obj:
                    raw_content = (raw_content + "\n/*RETRY*/\n" + retry_raw)[-12000:]

        if not obj:
            hint = ""
            if resp is not None and resp.status_code != 200:
                hint = f" HTTP {resp.status_code}"
            empty_note = "（模型返回正文为空）" if not (raw_content or "").strip() else ""
            return (
                None,
                f"[LLM] 无法从模型输出中解析出 verdict JSON{hint}{empty_note}；"
                f"请确认本机 Ollama 已拉取并运行 {model_name}，且未开启仅输出思考、截断过短等设置。"
                f" 原始片段: {raw_content[:240]}",
                "none",
                raw_content,
            )

        verdict = normalize_llm_verdict(str(obj.get("verdict", "")))
        if verdict not in ("合规", "违规", "违禁"):
            return None, str(obj.get("reason", "verdict 非法")), "none", raw_content

        reason = str(obj.get("reason", "")).strip() or "模型未给出理由"
        risk = str(obj.get("risk_level", "")).strip().lower()
        if risk not in ("high", "medium", "low", "none"):
            risk = "none" if verdict == "合规" else "medium"
        if _reason_is_echo_text(reason, text):
            reason = _fallback_reason(verdict, text, hits, variant_results)
        if len(reason) > 48:
            reason = reason[:48].rstrip() + "..."

        # 合规但命中强风险模式时复核一次，减少漏判
        if verdict == "合规" and AppConfig.LLM_ENABLE_SUSPECT_RECHECK:
            lur = live_high_risk_fallback(text)
            if lur:
                st2, rk2, desc2 = lur
                if st2 in ("违规", "违禁"):
                    verdict = st2
                    risk = rk2 if rk2 in ("high", "medium", "low", "none") else "medium"
                    reason = f"复核触发：{desc2}"

        # 纠偏：若模型判违规/违禁，但无词库命中且无高危规则命中，且语句明显中性描述，则回调为合规
        if verdict in ("违规", "违禁"):
            lur_now = live_high_risk_fallback(text)
            if (not hits) and (not variant_results) and (lur_now is None) and is_benign_product_description(text):
                verdict = "合规"
                risk = "none"
                reason = _compliant_short_reason(text)

        # 合规理由与原文类型不一致时（如服务话术却套「普通优惠」），用语义分桶短理由替换
        if verdict == "合规":
            if ("普通优惠" in reason or "优惠描述" in reason) and not re.search(
                r"宠粉价|折扣|优惠券|包邮|秒杀|立减|到手价|价|折|惠|元|块|给力|划算|活动",
                text or "",
            ):
                reason = _compliant_short_reason(text)
        return verdict, reason, risk, raw_content

    async def dispatch_audit(self, text: str, segment_id: str):
        """
        先推送 transcript_final（与后续 audit 共用 segment_id），再异步执行 LLM 审计。
        segment_id 须在 ASR completed 回调中通过 allocate_segment_id() 生成。
        """
        # 添加到上下文历史（仅保留最近5条）
        self.context_history.append(text)
        if len(self.context_history) > self.MAX_CONTEXT:
            self.context_history.pop(0)
        self._recent_utterances.append((text or "").strip())

        try:
            await self._emit_transcript_final(segment_id, text)
        except Exception as e:
            print(f"--- [转写终稿推送] {self.session_id}: {e} ---")
        if AppConfig.AUDIT_SEND_QUEUE_ACK:
            try:
                await self._send_result(
                    text,
                    "合规",
                    "none",
                    "[排队确认] 已进入终审队列，结果稍后更新",
                    "queue_ack",
                    "",
                    segment_id=segment_id,
                    llm_raw="",
                    msg_type="audit_fast",
                    is_final=False,
                )
            except Exception:
                pass
        # 入队给 worker（高风险优先）；若队列满则给出明确降级结果（不静默丢句）
        try:
            payload = (text, segment_id, time.time())
            if self._is_high_priority_text(text):
                self._audit_queue_hi.put_nowait(payload)
            else:
                self._audit_queue_norm.put_nowait(payload)
        except asyncio.QueueFull:
            # 高优先队列满时尝试降到普通队列一次
            try:
                self._audit_queue_norm.put_nowait((text, segment_id, time.time()))
                return
            except asyncio.QueueFull:
                pass
            hits = ac_match(text)
            ref_words = list(dict.fromkeys([h["word"] for h in hits if h.get("word")]))
            level = risk_level_from_hits(hits)
            await self._send_result(
                text,
                "违规" if hits else "合规",
                level if hits else "none",
                "[系统繁忙] 审计队列已满，当前句走拥塞降级；建议降低并发、提升算力或调小推流速率。",
                "system_busy",
                (hits[0]["word"] if hits else ""),
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw="",
            )

    async def drain_audits(self, timeout: Optional[float] = None) -> None:
        """等待所有在途审计任务结束（或超时），防止客户端已发 end 但 LLM 尚未返回。"""
        if timeout is None:
            timeout = AppConfig.AUDIT_DRAIN_TIMEOUT
        deadline = time.time() + timeout
        while time.time() < deadline:
            qsize = self._audit_backlog()
            inflight = self._audit_inflight
            if qsize == 0 and inflight == 0:
                await asyncio.sleep(0.4)
                if self._audit_backlog() == 0 and self._audit_inflight == 0:
                    return
            await asyncio.sleep(0.5)
        pend = self._audit_backlog() + self._audit_inflight
        print(
            f"--- [会话] {self.session_id} drain_audits 超时（>{timeout:.0f}s），"
            f"未完成审计任务数: {pend} ---"
        )

    async def _execute_audit(self, text: str, segment_id: str, *, queue_wait_ms: float = 0.0):
        """
        完整审计：词库/变体 → 仅作为 LLM 参考；终审由语言模型给出
        「合规 / 违规 / 违禁」。LLM 不可用时降级为原词库/变体直判（保留旧能力）。
        """
        self.last_activity = time.time()
        t0 = time.time()
        hits = ac_match(text)
        backlog = self._audit_backlog()
        variant_results: list[dict] = []
        if (not AppConfig.ENABLE_VARIANT_UNDER_LOAD) or (backlog < AppConfig.VARIANT_SKIP_BACKLOG):
            variant_results = sanitize_variant_results(list(variant_detector.detect(text) or []))
        ref_words = list(dict.fromkeys([h["word"] for h in hits if h.get("word")]))
        norm = normalize_text(text)[:512]
        combo = self._cross_sentence_violation(text) if AppConfig.ENABLE_CROSS_SENTENCE_RULE else None
        if (
            AppConfig.LLM_SKIP_LOWINFO_UTTERANCE
            and is_low_information_utterance(text)
            and (not hits)
            and (not variant_results)
            and (live_high_risk_fallback(text) is None)
        ):
            await self._send_result(
                text,
                "合规",
                "none",
                "[低信息短句] 口语停顿/填充词，不进入终审队列",
                "low_info_skip",
                "",
                segment_id=segment_id,
                hit_words=None,
                llm_raw="",
                perf_queue_ms=queue_wait_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
            )
            return

        # 命中同句缓存：减少重复请求 Ollama
        now = time.time()
        c = self._verdict_cache.get(norm)
        if c and (now - c[0] <= self._verdict_cache_ttl):
            st, rk, rsn, src = c[1], c[2], c[3], c[4]
            await self._send_result(
                text,
                st,
                rk,
                f"[缓存复用] {rsn}",
                src,
                "",
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw="",
            )
            if AppConfig.AUDIT_PERF_LOG:
                print(
                    f"--- [性能] {self.session_id} {segment_id} "
                    f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=cache ---"
                )
            return

        # 检查是否需要模型回退（仅针对非缓存、非跨句规则的句子）
        if not combo and not hits:
            fallback_key = f"{norm}_{segment_id}"
            if fallback_key not in _model_fallback_locks:
                _model_fallback_locks[fallback_key] = True
                model_name = llm_fallback_model_name()
                if model_name != llm_chat_model_name(False):
                    print(f"--- [模型回退] {self.session_id} {segment_id}: 主模型 {llm_chat_model_name(False)} 并发不足，回退到 {model_name}")
                    # 标记需要重试（但不影响当前结果）
                    import threading
                    self._model_fallback_signal = threading.Event()
                    self._model_fallback_signal.set()

        # 跨句组合风险：真实直播常分两句表达导流/功效承诺
        if combo:
            st_cb, rk_cb, rsn_cb = combo
            await self._send_result(
                text,
                st_cb,
                rk_cb,
                f"[跨句规则] {rsn_cb}",
                "cross_sentence_rule",
                "",
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw="",
                perf_queue_ms=queue_wait_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
            )
            self._verdict_cache[norm] = (time.time(), st_cb, rk_cb, rsn_cb, "cross_sentence_rule")
            if AppConfig.AUDIT_PERF_LOG:
                print(
                    f"--- [性能] {self.session_id} {segment_id} "
                    f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=cross_sentence_rule ---"
                )
            return

        # 通用直播高风险词前置直判：降低高并发下“后段高风险句”排队超时概率
        if (
            AppConfig.ENABLE_HIGHRISK_WORDLIST_PREFILTER
            and should_prefilter_highrisk_wordlist(text, hits)
        ):
            risk = risk_level_from_hits(hits)
            top_hit = max(
                hits,
                key=lambda h: {"high": 3, "medium": 2, "low": 1}.get(
                    str(h.get("level", "low")), 0
                ),
            )
            mw = str(top_hit.get("word", ""))
            await self._send_result(
                text,
                "违规",
                risk,
                f"[高风险词库前置] 命中高风险词“{mw}”，优先实时拦截",
                "highrisk_wordlist",
                mw,
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw="",
                perf_queue_ms=queue_wait_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
            )
            self._verdict_cache[norm] = (
                time.time(),
                "违规",
                risk,
                f"命中高风险词“{mw}”，优先实时拦截",
                "highrisk_wordlist",
            )
            if AppConfig.AUDIT_PERF_LOG:
                print(
                    f"--- [性能] {self.session_id} {segment_id} "
                    f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=highrisk_wordlist ---"
                )
            return

        # 极高置信规则预筛：命中则跳过 LLM，降低时延并减少典型漏判（刷单/私信留电话等）
        if AppConfig.LLM_ENABLE_HIGHCONF_RULE_PREFILTER:
            lur_pf = live_high_risk_fallback(text)
            if lur_pf:
                st_pf, rk_pf, desc_pf = lur_pf
                await self._send_result(
                    text,
                    st_pf,
                    rk_pf,
                    f"[快速规则] {desc_pf}",
                    "highconf_rule",
                    "",
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                self._verdict_cache[norm] = (time.time(), st_pf, rk_pf, desc_pf, "highconf_rule")
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=highconf_rule ---"
                    )
                return

        # 高置信合规预筛：中性描述且无命中时直接合规，减少无必要 LLM 调用
        if (
            AppConfig.LLM_ENABLE_BENIGN_PREFILTER
            and (not hits)
            and (not variant_results)
            and is_benign_product_description(text)
        ):
            rsn = _compliant_short_reason(text)
            await self._send_result(
                text,
                "合规",
                "none",
                f"[快速规则] {rsn}",
                "benign_rule",
                "",
                segment_id=segment_id,
                hit_words=None,
                llm_raw="",
                perf_queue_ms=queue_wait_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
            )
            self._verdict_cache[norm] = (time.time(), "合规", "none", rsn, "benign_rule")
            if AppConfig.AUDIT_PERF_LOG:
                print(
                    f"--- [性能] {self.session_id} {segment_id} "
                    f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=benign_rule ---"
                )
            return

        # 劝导/辟谣/警示：即使命中词库也优先合规（避免排队预算/词库降级误杀）
        if hits and is_compliance_advisory_tone(text) and live_high_risk_fallback(text) is None:
            await self._send_result(
                text,
                "合规",
                "none",
                "[快速规则] 劝导/辟谣/警示语境，命中词不按违规处理",
                "advisory_compliant",
                "",
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw="",
                perf_queue_ms=queue_wait_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
            )
            self._verdict_cache[norm] = (
                time.time(),
                "合规",
                "none",
                "劝导/辟谣/警示语境，命中词不按违规处理",
                "advisory_compliant",
            )
            if AppConfig.AUDIT_PERF_LOG:
                print(
                    f"--- [性能] {self.session_id} {segment_id} "
                    f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=advisory_compliant ---"
                )
            return

        # 高置信快速直判（保守触发）：默认不抢在 LLM 前直判，避免影响准确率。
        # 仅在启用开关且会话队列明显拥塞时，才用快速规则做实时兜底。
        queue_busy = self._audit_backlog() >= max(3, AppConfig.MAX_QUEUE_SIZE // 5)
        if AppConfig.LLM_ENABLE_RULE_SHORTCUT and queue_busy:
            lur = live_high_risk_fallback(text)
            if lur:
                st, rk, desc = lur
                await self._send_result(
                    text,
                    st,
                    rk,
                    f"[快速规则] {desc}",
                    "fast_rule",
                    "",
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                self._verdict_cache[norm] = (time.time(), st, rk, desc, "fast_rule")
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=fast_rule ---"
                    )
                return
            if hits and is_compliance_advisory_tone(text):
                await self._send_result(
                    text,
                    "合规",
                    "none",
                    "[快速规则] 劝导/辟谣/警示语境，命中词不按违规处理",
                    "fast_rule",
                    "",
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                )
                self._verdict_cache[norm] = (
                    time.time(),
                    "合规",
                    "none",
                    "劝导/辟谣/警示语境，命中词不按违规处理",
                    "fast_rule",
                )
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=fast_rule ---"
                    )
                return

        # 常态排队预算保护：队列等待过久时优先可解释快出，避免后段句子持续堆积后超时
        if AppConfig.ENABLE_QUEUE_BUDGET_DEGRADE and queue_wait_ms >= AppConfig.AUDIT_QUEUE_BUDGET_MS:
            if hits and is_compliance_advisory_tone(text) and live_high_risk_fallback(text) is None:
                await self._send_result(
                    text,
                    "合规",
                    "none",
                    f"[排队预算] 排队较久({queue_wait_ms:.0f}ms)但属劝导/辟谣语境，不因命中词降级为违规",
                    "queue_budget_advisory",
                    "",
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=queue_budget_advisory ---"
                    )
                return
            if hits:
                level = risk_level_from_hits(hits)
                mw = str(hits[0].get("word", "")) if hits else ""
                await self._send_result(
                    text,
                    "违规",
                    level,
                    f"[排队预算-词库] 排队较久({queue_wait_ms:.0f}ms)，优先按命中线索给出实时判定",
                    "queue_budget_wordlist",
                    mw,
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=queue_budget_wordlist ---"
                    )
                return
            if variant_results:
                vr = variant_results[0]
                await self._send_result(
                    text,
                    "违规",
                    "medium",
                    f"[排队预算-变体] 排队较久({queue_wait_ms:.0f}ms)，疑似绕过: "
                    f"{vr.get('variant', '')} -> {vr.get('original', '')}",
                    "queue_budget_variant",
                    str(vr.get("original", "")),
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=queue_budget_variant ---"
                    )
                return

        # 双阶段输出：对需要走 LLM 的句子先给快判，再给终审
        if AppConfig.ENABLE_DUAL_STAGE_AUDIT:
            fp_st, fp_rk, fp_reason = self._quick_preview_verdict(text, hits, variant_results)
            await self._send_result(
                text,
                fp_st,
                fp_rk,
                fp_reason,
                "fast_preview",
                "",
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw="",
                msg_type="audit_fast",
                is_final=False,
                perf_queue_ms=queue_wait_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
            )

        # 收尾阶段（已收到 end）若排队已久，优先保时延：对有命中线索句子走可解释降级
        if (
            AppConfig.ENABLE_TAIL_FAST_DEGRADE
            and self._audio_ended
            and queue_wait_ms >= AppConfig.TAIL_FAST_QUEUE_WAIT_MS
        ):
            if hits and is_compliance_advisory_tone(text) and live_high_risk_fallback(text) is None:
                await self._send_result(
                    text,
                    "合规",
                    "none",
                    f"[收尾加速] 结束后排队较久({queue_wait_ms:.0f}ms)但属劝导/辟谣语境，不因命中词降级为违规",
                    "tail_fast_advisory",
                    "",
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=tail_fast_advisory ---"
                    )
                return
            if hits:
                level = risk_level_from_hits(hits)
                mw = str(hits[0].get("word", "")) if hits else ""
                await self._send_result(
                    text,
                    "违规",
                    level,
                    f"[收尾加速-词库] 结束后在途排队较久({queue_wait_ms:.0f}ms)，优先按命中线索给出实时判定",
                    "tail_fast_wordlist",
                    mw,
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=tail_fast_wordlist ---"
                    )
                return
            if variant_results:
                vr = variant_results[0]
                await self._send_result(
                    text,
                    "违规",
                    "medium",
                    f"[收尾加速-变体] 结束后在途排队较久({queue_wait_ms:.0f}ms)，疑似绕过: "
                    f"{vr.get('variant', '')} -> {vr.get('original', '')}",
                    "tail_fast_variant",
                    str(vr.get("original", "")),
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw="",
                    perf_queue_ms=queue_wait_ms,
                    perf_total_ms=(time.time() - t0) * 1000.0,
                )
                if AppConfig.AUDIT_PERF_LOG:
                    print(
                        f"--- [性能] {self.session_id} {segment_id} "
                        f"queue={queue_wait_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=tail_fast_variant ---"
                    )
                return

        llm_t0 = time.time()
        sem_wait_ms = 0.0
        infer_ms = 0.0
        verdict: Optional[str] = None
        reason_text = ""
        risk_llm = ""
        raw_llm = ""

        # 检查是否需要模型回退（仅对非缓存、非跨句规则的句子）
        use_fallback = should_use_fallback_model(text, segment_id)
        model_for_this_request = llm_fallback_model_name() if use_fallback else None
        if use_fallback:
            print(f"--- [模型回退] {self.session_id} {segment_id}: 使用备用模型 {model_for_this_request}")

        if AppConfig.LLM_AUDIT_SINGLE_CALL:
            # 精确率优先：单次完整终审，避免「快判超时 + 复核再超时」双段丢模与词库误降级
            inflight_key = norm + f"#full_only_fallback={use_fallback}"
            task = self._llm_inflight.get(inflight_key)
            owner = False
            if task is None:
                task = asyncio.create_task(
                    self._llm_semantic_verdict(
                        text, hits, variant_results, fast_mode=False, model_override=model_for_this_request
                    )
                )
                self._llm_inflight[inflight_key] = task
                owner = True
            try:
                verdict, reason_text, risk_llm, raw_llm, sem_wait_ms, infer_ms = await task
            finally:
                if owner and self._llm_inflight.get(inflight_key) is task:
                    self._llm_inflight.pop(inflight_key, None)
        else:
            inflight_key = norm + f"#fast_fallback={use_fallback}"
            task = self._llm_inflight.get(inflight_key)
            owner = False
            if task is None:
                task = asyncio.create_task(
                    self._llm_semantic_verdict(text, hits, variant_results, fast_mode=True, model_override=model_for_this_request)
                )
                self._llm_inflight[inflight_key] = task
                owner = True
            try:
                verdict, reason_text, risk_llm, raw_llm, sem_wait_ms, infer_ms = await task
            finally:
                if owner and self._llm_inflight.get(inflight_key) is task:
                    self._llm_inflight.pop(inflight_key, None)

        # 快判冲突/低置信时，再走一次终审复核（full prompt）
        if (
            not AppConfig.LLM_AUDIT_SINGLE_CALL
            and AppConfig.ENABLE_CONFLICT_RECHECK
            and self._need_recheck(
                verdict,
                text,
                reason_text,
                hits,
                variant_results,
                backlog=backlog,
                queue_wait_ms=queue_wait_ms,
            )
        ):
            inflight_key2 = norm + f"#full_recheck_fallback={use_fallback}"
            task2 = self._llm_inflight.get(inflight_key2)
            owner2 = False
            if task2 is None:
                task2 = asyncio.create_task(
                    self._llm_semantic_verdict(
                        text, hits, variant_results, fast_mode=False, model_override=model_for_this_request
                    )
                )
                self._llm_inflight[inflight_key2] = task2
                owner2 = True
            try:
                v2, r2, rk2, raw2, sw2, inf2 = await task2
            finally:
                if owner2 and self._llm_inflight.get(inflight_key2) is task2:
                    self._llm_inflight.pop(inflight_key2, None)
            if v2 is not None:
                verdict, reason_text, risk_llm, raw_llm = v2, r2, rk2, raw2
                sem_wait_ms += sw2
                infer_ms += inf2

        llm_ms = (time.time() - llm_t0) * 1000.0

        if verdict is None:
            # 降级：LLM 失败时不能简单「命中即违规」，先用语境启发式减少误杀
            if hits and is_compliance_advisory_tone(text):
                await self._send_result(
                    text,
                    "合规",
                    "none",
                    f"[降级-语境规则] 命中敏感词但表述为劝导/禁止或辟谣否定类；"
                    f"LLM 终审链路未生效（快判模型 {llm_chat_model_name(True)}，终审模型 {AppConfig.OLLAMA_MODEL}）：{reason_text}",
                    "system",
                    "",
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw=raw_llm,
                )
                return

            # 直播高危话术补判（词库未覆盖或 LLM 未生效时减少语义漏判）
            lur = live_high_risk_fallback(text)
            if lur:
                st, rk, desc = lur
                await self._send_result(
                    text,
                    st,
                    rk,
                    f"[降级-直播风控] {desc}；{reason_text}",
                    "live_risk",
                    "",
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw=raw_llm,
                )
                return

            if hits:
                risk_order = {"high": 3, "medium": 2, "low": 1}
                top_hit = max(hits, key=lambda h: risk_order.get(h.get("level", "low"), 0))
                variant_info = ""
                if "variant" in top_hit:
                    variant_info = f"（变体: {top_hit['variant']}）"
                matched_word = top_hit["word"]
                level = top_hit.get("level", "high")
                await self._send_result(
                    text,
                    "违规",
                    level,
                    f"[降级-词库] 命中严禁词: {matched_word}{variant_info}；{reason_text}",
                    "word_list",
                    matched_word,
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw=raw_llm,
                )
                return
            if variant_results:
                vr = variant_results[0]
                await self._send_result(
                    text,
                    "违规",
                    "medium",
                    f"[降级-变体] 疑似绕过: {vr.get('variant', '')} → {vr.get('original', '')} ({vr.get('method', '')})；{reason_text}",
                    "variant",
                    str(vr.get("original", "")),
                    segment_id=segment_id,
                    hit_words=ref_words or None,
                    llm_raw=raw_llm,
                )
                return
            await self._send_result(
                text,
                "合规",
                "none",
                f"[降级] {reason_text}",
                "system",
                "",
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw=raw_llm,
                perf_queue_ms=queue_wait_ms,
                perf_llm_ms=llm_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
                perf_sem_wait_ms=sem_wait_ms,
                perf_infer_ms=infer_ms,
            )
            return

        if verdict == "合规":
            await self._send_result(
                text,
                "合规",
                risk_llm or "none",
                f"[LLM] {reason_text}",
                "llm",
                "",
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw=raw_llm,
                perf_queue_ms=queue_wait_ms,
                perf_llm_ms=llm_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
                perf_sem_wait_ms=sem_wait_ms,
                perf_infer_ms=infer_ms,
            )
            self._verdict_cache[norm] = (
                time.time(),
                "合规",
                (risk_llm or "none"),
                reason_text,
                "llm",
            )
            if AppConfig.AUDIT_PERF_LOG:
                print(
                    f"--- [性能] {self.session_id} {segment_id} "
                    f"queue={queue_wait_ms:.0f}ms llm={llm_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=llm ---"
                )
            return

        if verdict == "违禁":
            await self._send_result(
                text,
                "违禁",
                risk_llm or "medium",
                f"[LLM] {reason_text}",
                "llm",
                "",
                segment_id=segment_id,
                hit_words=ref_words or None,
                llm_raw=raw_llm,
                perf_queue_ms=queue_wait_ms,
                perf_llm_ms=llm_ms,
                perf_total_ms=(time.time() - t0) * 1000.0,
                perf_sem_wait_ms=sem_wait_ms,
                perf_infer_ms=infer_ms,
            )
            self._verdict_cache[norm] = (
                time.time(),
                "违禁",
                (risk_llm or "medium"),
                reason_text,
                "llm",
            )
            if AppConfig.AUDIT_PERF_LOG:
                print(
                    f"--- [性能] {self.session_id} {segment_id} "
                    f"queue={queue_wait_ms:.0f}ms llm={llm_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=llm ---"
                )
            return

        # 违规：尽量带上主命中词便于落库与统计
        matched_for_db = ""
        if hits:
            risk_order = {"high": 3, "medium": 2, "low": 1}
            top_hit = max(hits, key=lambda h: risk_order.get(h.get("level", "low"), 0))
            matched_for_db = str(top_hit.get("word", ""))
        elif variant_results:
            matched_for_db = str(variant_results[0].get("original", ""))

        rk = risk_llm if risk_llm in ("high", "medium", "low", "none") else risk_level_from_hits(hits)
        await self._send_result(
            text,
            "违规",
            rk,
            f"[LLM] {reason_text}",
            "llm",
            matched_for_db,
            segment_id=segment_id,
            hit_words=ref_words or None,
            llm_raw=raw_llm,
            perf_queue_ms=queue_wait_ms,
            perf_llm_ms=llm_ms,
            perf_total_ms=(time.time() - t0) * 1000.0,
            perf_sem_wait_ms=sem_wait_ms,
            perf_infer_ms=infer_ms,
        )
        self._verdict_cache[norm] = (time.time(), "违规", rk, reason_text, "llm")
        if AppConfig.AUDIT_PERF_LOG:
            print(
                f"--- [性能] {self.session_id} {segment_id} "
                f"queue={queue_wait_ms:.0f}ms llm={llm_ms:.0f}ms total={(time.time() - t0)*1000:.0f}ms src=llm ---"
            )

    async def _send_result(
        self,
        text: str,
        status: str,
        risk_level: str,
        reason: str,
        source: str,
        matched_word: str,
        *,
        segment_id: str = "",
        hit_words: Optional[list[str]] = None,
        llm_raw: str = "",
        msg_type: str = "audit",
        is_final: bool = True,
        perf_queue_ms: Optional[float] = None,
        perf_llm_ms: Optional[float] = None,
        perf_total_ms: Optional[float] = None,
        perf_sem_wait_ms: Optional[float] = None,
        perf_infer_ms: Optional[float] = None,
    ):
        """发送审计结果并持久化"""
        if not self.is_active:
            return

        if status == "合规":
            risk_level = "none"
        elif status == "违禁" and (not risk_level or risk_level == "none"):
            risk_level = "medium"

        # 记录到数据库
        await log_audit(
            self.session_id,
            self.room_id,
            text,
            status,
            risk_level,
            source,
            matched_word,
            segment_id=segment_id or "",
        )

        # 实时告警（异步，不阻塞主流程）
        asyncio.create_task(
            send_alert_webhook(
                self.room_id,
                text,
                status,
                risk_level,
                reason,
                matched_word,
                self.session_id,
            )
        )

        # 推送到 WebSocket
        try:
            out_type = msg_type
            if AppConfig.ENABLE_DUAL_STAGE_AUDIT and msg_type == "audit" and is_final:
                out_type = "audit_final"
            payload = {
                "type": out_type,
                "is_final": bool(is_final),
                "segment_id": segment_id or "",
                "text": text,
                "status": status,
                "risk_level": risk_level,
                "reason": reason,
                "source": source,
                "session_id": self.session_id,
                "room_id": self.room_id,
                "timestamp": datetime.now().isoformat(),
            }
            perf = {}
            if perf_queue_ms is not None:
                perf["queue_ms"] = float(perf_queue_ms)
            if perf_llm_ms is not None:
                perf["llm_ms"] = float(perf_llm_ms)
            if perf_total_ms is not None:
                total_ms = float(perf_total_ms)
                if perf_queue_ms is not None:
                    qv = float(perf_queue_ms)
                    # total 统一按「入队到出结果」口径；若上传的是执行段耗时，补上排队时间。
                    if total_ms < qv:
                        total_ms = qv + max(0.0, total_ms)
                perf["total_ms"] = total_ms
            if perf_sem_wait_ms is not None:
                perf["sem_wait_ms"] = float(perf_sem_wait_ms)
            if perf_infer_ms is not None:
                perf["infer_ms"] = float(perf_infer_ms)
            if perf:
                payload["perf"] = perf
            if hit_words:
                payload["hit_words"] = hit_words
            if llm_raw:
                payload["llm_raw"] = str(llm_raw)[:4000]
            await self.ws.send_json(payload)
        except Exception:
            pass

    async def _heartbeat_loop(self):
        """心跳保活：定期发送 ping，检测连接状态"""
        try:
            while self.is_active:
                await asyncio.sleep(AppConfig.HEARTBEAT_INTERVAL)
                if self.ws.client_state.value == 1:  # CONNECTED
                    await self.ws.send_json({"type": "heartbeat", "session_id": self.session_id})
                    # 有在途审计时勿因「无 ASR 活动」误判超时（LLM 耗时不写入 partial）
                    if self._audit_inflight > 0 or self._audit_backlog() > 0:
                        continue
                    # 检查是否超时
                    if time.time() - self.last_activity > AppConfig.HEARTBEAT_TIMEOUT:
                        print(f"--- [心跳超时] {self.session_id} 超过 {AppConfig.HEARTBEAT_TIMEOUT}s 无活动 ---")
                        self.is_active = False
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"--- [心跳异常] {self.session_id}: {e} ---")
            self.is_active = False

    async def _asr_keepalive_loop(self) -> None:
        """定期向 DashScope 追加极短 PCM 静音，降低上游因长时间无有效音频而断连的概率。"""
        samples = max(
            1,
            int(AppConfig.TARGET_RATE * (AppConfig.ASR_KEEPALIVE_PCM_MS / 1000.0)),
        )
        silence_b64 = base64.b64encode(
            np.zeros(samples, dtype=np.int16).tobytes()
        ).decode("ascii")
        try:
            while self.is_active:
                await asyncio.sleep(AppConfig.ASR_KEEPALIVE_INTERVAL_SEC)
                if not self.is_active or self._audio_ended:
                    continue
                conv = self.conv
                if conv is None:
                    continue
                try:
                    conv.append_audio(silence_b64)
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass

    async def close(self):
        """资源释放：先等审计跑完，再关 ASR，避免测试脚本统计审计句数为 0"""
        if hasattr(self, "heartbeat_task"):
            self.heartbeat_task.cancel()
        if self._asr_keepalive_task:
            self._asr_keepalive_task.cancel()
            await asyncio.gather(self._asr_keepalive_task, return_exceptions=True)
        await self.drain_audits(timeout=AppConfig.AUDIT_DRAIN_TIMEOUT)
        self.is_active = False
        # 停掉审计 worker，避免关闭 http_client 后仍有任务试图请求 Ollama
        for wt in self._audit_workers:
            wt.cancel()
        if self._audit_workers:
            await asyncio.gather(*self._audit_workers, return_exceptions=True)
        await self.http_client.aclose()
        if self.conv:
            self.conv.close()
        await end_session(self.session_id)
        active_sessions.pop(self.session_id, None)
        print(f"--- [会话] {self.session_id} 已关闭 ---")


# ================= Lifespan 生命周期 =================


class _RedactAccessLogApiKeyFilter(logging.Filter):
    """掩码访问日志里的 `?api_key=` / `&api_key=`，避免 Uvicorn 把客户端密钥打到终端。"""

    _pat = re.compile(r"([?&])api_key=[^&\s\"']+", re.I)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = self._pat.sub(r"\1api_key=***", record.msg)
            if record.args:
                record.args = tuple(
                    self._pat.sub(r"\1api_key=***", a) if isinstance(a, str) else a
                    for a in record.args
                )
        except Exception:
            pass
        return True


def _install_access_log_redact_filter() -> None:
    if os.getenv("ACCESS_LOG_REDACT_API_KEY", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return
    for name in ("uvicorn.access",):
        logging.getLogger(name).addFilter(_RedactAccessLogApiKeyFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _app_loop
    _install_access_log_redact_filter()
    print("=" * 50)
    print("  直播间语音识别与审计系统 v2.0")
    print("=" * 50)
    _dk = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    print(
        "--- [DashScope] ASR 模型="
        f"{AppConfig.DASHSCOPE_ASR_MODEL}, 实时 WS="
        f"{AppConfig.DASHSCOPE_REALTIME_WS_URL or '默认(北京 dashscope.aliyuncs.com)'}, "
        f"API_KEY={'已配置' if _dk else '未配置'} ---"
    )
    await init_db()
    _app_loop = asyncio.get_running_loop()
    await rebuild_ac_automaton()

    # Ollama 预热：避免首次请求触发冷启动导致首句 LLM 推理显著变慢
    if os.getenv("OLLAMA_WARMUP_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            warm_num_ctx = int(os.getenv("OLLAMA_WARMUP_NUM_CTX", "256"))
            warm_num_predict = int(os.getenv("OLLAMA_WARMUP_NUM_PREDICT", "32"))
            warm_timeout = float(os.getenv("OLLAMA_WARMUP_TIMEOUT_SEC", "10"))
            warm_model = AppConfig.OLLAMA_MODEL
            warm_payload = {
                "model": warm_model,
                "messages": [
                    {"role": "system", "content": "输出一个空 JSON：{\"verdict\":\"合规\",\"reason\":\"预热\",\"risk_level\":\"none\"}"},
                    {"role": "user", "content": "预热"},
                ],
                "stream": False,
                "options": {
                    "num_ctx": warm_num_ctx,
                    "temperature": 0.01,
                    "num_predict": warm_num_predict,
                    "top_p": AppConfig.OLLAMA_TOP_P,
                    "repeat_penalty": AppConfig.OLLAMA_REPEAT_PENALTY,
                },
            }
            import httpx as _httpx  # 已在模块顶部引入，但此处确保不被重命名覆盖
            async with _httpx.AsyncClient(timeout=warm_timeout) as _client:
                await _client.post(AppConfig.OLLAMA_CHAT_URL, json=warm_payload)
            print("--- [Ollama] warmup done ---")
        except Exception as e:
            print(f"--- [Ollama] warmup failed: {type(e).__name__}: {e} ---")

    observer = start_yjc_watcher()
    yield
    # Shutdown
    print("--- [系统] 服务正在关闭 ---")
    for sid, mgr in list(active_sessions.items()):
        await mgr.close()
    if observer:
        observer.stop()
        observer.join()


app = FastAPI(title="直播语音审计系统", version="2.0", lifespan=lifespan)


# ================= WebSocket 路由 =================

def _ws_client_auth_key(websocket: WebSocket, query_api_key: Optional[str]) -> Optional[str]:
    """
    客户端连接 /ws/audit 时的身份密钥。优先从 WebSocket 握手头读取，避免 ?api_key= 进入 Uvicorn access log。
    顺序：Authorization: Bearer <token> → X-API-Key →（可选）查询参数 api_key。
    """
    h = websocket.headers
    auth = (h.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        v = auth[7:].strip()
        if v:
            return v
    x = (h.get("x-api-key") or "").strip()
    if x:
        return x
    if AppConfig.AUTH_WS_ALLOW_QUERY_KEY:
        q = (query_api_key or "").strip()
        return q if q else None
    return None


def _dashscope_asr_socket_gone(exc: BaseException) -> bool:
    """
    DashScope 实时 ASR（omni_realtime）底层使用 websocket-client；
    云端断线、会话超时、配额/鉴权踢线后仍 append_audio 会抛出「连接已关闭」类异常。
    """
    name = type(exc).__name__
    if "WebSocketConnectionClosed" in name or "ConnectionClosedError" in name:
        return True
    if isinstance(exc, TimeoutError):
        return True
    s = str(exc).lower()
    return (
        "already closed" in s
        or "connection is already closed" in s
        or "websocket closed due to" in s
        or "could not established within" in s
        or "broken pipe" in s
        or "connection reset" in s
    )


@app.websocket("/ws/audit")
async def audit_websocket_endpoint(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None),
    room_id_query: Optional[str] = Query(None),
):
    # 认证校验（密钥勿放 URL 查询串，见 _ws_client_auth_key）
    client_key = _ws_client_auth_key(websocket, api_key)
    if AppConfig.AUTH_ENABLED and AppConfig.AUTH_API_KEY:
        if client_key != AppConfig.AUTH_API_KEY:
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "code": "auth_failed",
                "message": "认证失败：API Key 不正确"
            })
            await websocket.close(code=1008)
            return

    await websocket.accept()

    room_id = room_id_query or ""
    client_ip = websocket.client.host if websocket.client else "unknown"

    stream_processor = AudioStreamProcessor()
    audit_manager = None
    # DashScope 上游 WebSocket 已死：不再 append_audio，避免刷屏异常与对客户端重复 error
    dashscope_audio_dead = False
    dashscope_dead_error_sent = False
    asr_fail_window_start = 0.0
    asr_fail_count = 0
    asr_pause_until = 0.0

    try:
        while True:
            message = await asyncio.wait_for(
                websocket.receive(),
                timeout=AppConfig.AUDIO_TIMEOUT
            )

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("type") == "websocket.receive":
                # 处理二进制音频数据
                if message.get("bytes"):
                    if time.time() < asr_pause_until:
                        continue
                    if audit_manager is None:
                        # 首次收到音频时创建审计管理器（延迟初始化）
                        # DashScope connect/update_session 失败时若未捕获，会导致 TCP 被硬断，
                        # 客户端报「no close frame received or sent」。
                        try:
                            async with _session_admission_lock:
                                if len(active_sessions) >= AppConfig.MAX_CONCURRENT:
                                    await websocket.send_json(
                                        {
                                            "type": "error",
                                            "code": "capacity_full",
                                            "message": (
                                                f"当前并发直播路数已达上限（{AppConfig.MAX_CONCURRENT}），"
                                                "请稍后重试或增大环境变量 WS_MAX_SESSIONS。"
                                            )[:2000],
                                            "room_id": room_id,
                                        }
                                    )
                                    await websocket.close(code=1008)
                                    return
                                audit_manager = LiveAuditManager(
                                    websocket, room_id, client_ip
                                )
                                active_sessions[audit_manager.session_id] = audit_manager
                        except Exception as e:
                            print(
                                f"--- [ASR 初始化失败] {room_id}/{client_ip}: {type(e).__name__}: {e} ---"
                            )
                            traceback.print_exc()
                            try:
                                await websocket.send_json(
                                    {
                                        "type": "error",
                                        "code": "asr_init_failed",
                                        "message": (
                                            "ASR（DashScope 实时转写）初始化失败。"
                                            "请检查 apikey.env 中 DASHSCOPE_API_KEY、DASHSCOPE_ASR_MODEL、本机出网、"
                                            f"dashscope SDK 版本，以及模型名是否与控制台一致（当前主模型 {AppConfig.DASHSCOPE_ASR_MODEL}）。"
                                            f" 详情: {e}"
                                        )[:2000],
                                        "room_id": room_id,
                                    }
                                )
                            except Exception:
                                pass
                            try:
                                await websocket.close(code=1011)
                            except Exception:
                                pass
                            return

                    try:
                        processed_pcm, success = stream_processor.process(
                            message["bytes"]
                        )
                        if (
                            success
                            and audit_manager
                            and audit_manager.conv
                        ):
                            if dashscope_audio_dead:
                                recovered = await audit_manager.reconnect_qwen_asr(
                                    reason="resume_after_upstream_closed"
                                )
                                dashscope_audio_dead = not recovered
                            if dashscope_audio_dead:
                                continue
                            b64_data = base64.b64encode(processed_pcm).decode("ascii")
                            audit_manager.conv.append_audio(b64_data)
                            if asr_fail_count > 0:
                                asr_fail_count = 0
                                asr_fail_window_start = 0.0
                                dashscope_dead_error_sent = False
                    except Exception as e:
                        if _dashscope_asr_socket_gone(e):
                            recovered = False
                            if audit_manager:
                                recovered = await audit_manager.reconnect_qwen_asr(
                                    reason="append_audio_closed"
                                )
                            if recovered and audit_manager and audit_manager.conv:
                                try:
                                    audit_manager.conv.append_audio(
                                        base64.b64encode(processed_pcm).decode("ascii")
                                    )
                                    dashscope_audio_dead = False
                                    asr_fail_count = 0
                                    asr_fail_window_start = 0.0
                                    dashscope_dead_error_sent = False
                                    continue
                                except Exception as retry_e:
                                    if not _dashscope_asr_socket_gone(retry_e):
                                        raise
                            dashscope_audio_dead = True
                            now = time.time()
                            if (
                                asr_fail_window_start <= 0
                                or (now - asr_fail_window_start) > AppConfig.ASR_UPSTREAM_FATAL_FAIL_WINDOW_SEC
                            ):
                                asr_fail_window_start = now
                                asr_fail_count = 0
                            asr_fail_count += 1
                            if asr_fail_count < AppConfig.ASR_UPSTREAM_FATAL_FAIL_THRESHOLD:
                                print(
                                    f"--- [ASR 断链重试中] {room_id}/{client_ip} "
                                    f"fail={asr_fail_count}/{AppConfig.ASR_UPSTREAM_FATAL_FAIL_THRESHOLD} "
                                    f"window={AppConfig.ASR_UPSTREAM_FATAL_FAIL_WINDOW_SEC:.0f}s ---"
                                )
                                continue
                            asr_pause_until = time.time() + max(0.0, AppConfig.ASR_UPSTREAM_PAUSE_SEC)
                            print(
                                f"--- [ASR 熔断暂停] {room_id}/{client_ip} "
                                f"达到失败阈值，暂停 {AppConfig.ASR_UPSTREAM_PAUSE_SEC:.0f}s 后再试 ---"
                            )
                            if not dashscope_dead_error_sent:
                                dashscope_dead_error_sent = True
                                print(
                                    f"--- [ASR 上游已断开] {room_id}/{client_ip} "
                                    f"DashScope 实时 WebSocket 已关闭，后续音频不再投递。"
                                    f" 常见原因：云端会话超时/网络闪断/密钥配额或鉴权。"
                                    f" 异常: {type(e).__name__}: {e} ---"
                                )
                                try:
                                    await websocket.send_json(
                                        {
                                            "type": "error",
                                            "code": "asr_upstream_closed",
                                            "message": (
                                                "阿里云 DashScope 实时 ASR 连接已断开，本路无法再上传音频。"
                                                "已自动重连多次仍失败，请检查 DASHSCOPE_API_KEY、配额与本机出网；"
                                                "若 Key 在国际（新加坡）控制台开通，请在 apikey.env 设置 "
                                                "DASHSCOPE_REALTIME_WS_URL=wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime "
                                                "后重启服务。"
                                            )[:2000],
                                            "room_id": room_id,
                                        }
                                    )
                                except Exception:
                                    pass
                            continue
                        print(
                            f"--- [音频/ASR 推送异常] {room_id}/{client_ip}: "
                            f"{type(e).__name__}: {e} ---"
                        )
                        traceback.print_exc()
                        try:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "code": "audio_pipeline_error",
                                    "message": str(e)[:2000],
                                    "room_id": room_id,
                                }
                            )
                        except Exception:
                            pass

                # 处理文本消息（元数据、心跳响应等）
                elif message.get("text"):
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type")

                        if msg_type == "meta":
                            stream_processor.set_input_rate(data.get("sample_rate", 16000))
                            room_id = data.get("room_id", room_id)
                            if audit_manager:
                                audit_manager.room_id = room_id
                            print(f"--- [元数据] 采样率: {data.get('sample_rate')}, 房间: {room_id} ---")

                        elif msg_type == "pong":
                            # 客户端心跳响应
                            if audit_manager:
                                audit_manager.last_activity = time.time()

                        elif msg_type == "ping":
                            # 客户端主动 ping，服务端回应 pong
                            await websocket.send_json({"type": "pong"})

                        elif msg_type == "end":
                            # 客户端声明音频结束：用多段静音尾推 ASR 出 completed，再稍等避免立即断连
                            if audit_manager:
                                audit_manager.last_activity = time.time()
                                audit_manager.mark_audio_ended()
                            if (
                                audit_manager
                                and audit_manager.conv
                                and stream_processor
                                and not dashscope_audio_dead
                            ):
                                for _ in range(4):
                                    tail = np.zeros(
                                        int(AppConfig.TARGET_RATE * 0.35), dtype=np.int16
                                    ).tobytes()
                                    processed_pcm, ok = stream_processor.process(tail)
                                    if ok:
                                        try:
                                            audit_manager.conv.append_audio(
                                                base64.b64encode(
                                                    processed_pcm
                                                ).decode("ascii")
                                            )
                                        except Exception as tail_e:
                                            if _dashscope_asr_socket_gone(tail_e):
                                                dashscope_audio_dead = True
                                                if audit_manager:
                                                    audit_manager.is_active = False
                                            break
                                    await asyncio.sleep(0.05)
                            await asyncio.sleep(1.5)
                            break

                    except json.JSONDecodeError:
                        pass

    except WebSocketDisconnect:
        print(f"--- [连接断开] {room_id}/{client_ip} ---")
    except asyncio.TimeoutError:
        print(f"--- [超时断开] {room_id}/{client_ip} 超过 {AppConfig.AUDIO_TIMEOUT}s 无数据 ---")
    except Exception as e:
        print(f"--- [异常] {room_id}: {e} ---")
    finally:
        if audit_manager:
            await audit_manager.close()
        print(f"--- [清理] {room_id} 会话资源已释放 ---")


# ================= REST 管理 API =================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "max_sessions": AppConfig.MAX_CONCURRENT,
        "ollama_global_max_concurrent": AppConfig.OLLAMA_GLOBAL_MAX_CONCURRENT,
        "uptime": time.time(),
        "version": "2.1",
    }


@app.get("/api/sessions")
async def list_sessions():
    """列出当前活跃会话"""
    return {"sessions": list(active_sessions.values())}


@app.post("/api/words/reload")
async def reload_words():
    """手动触发违禁词热重载"""
    try:
        await rebuild_ac_automaton()
        return {"status": "ok", "message": "词库已重新加载"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/words/upload")
async def upload_words(request: Request):
    """
    上传违禁词表（JSON 格式）
    Body: {"words": [{"word": "xxx", "level": "high", "category": "广告", "note": ""}, ...]}
    同时也会覆盖 yjc.txt 文件
    """
    try:
        body = await request.json()
        words = body.get("words", [])
        if not words:
            return JSONResponse(status_code=400, content={"status": "error", "message": "缺少 words 字段"})

        # 写入 yjc.txt
        lines = []
        for w in words:
            line = f"{w['word']}|{w.get('level', 'high')}|{w.get('category', '')}|{w.get('note', '')}"
            lines.append(line)
        YJC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 重新加载
        await rebuild_ac_automaton()
        return {"status": "ok", "message": f"已上传并加载 {len(words)} 条违禁词"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/logs/violations")
async def get_violations(limit: int = Query(50, le=500)):
    """查询最近违规记录"""
    violations = await get_recent_violations(limit)
    return {"total": len(violations), "violations": violations}


@app.get("/api/stats")
async def stats(room_id: str = Query(None)):
    """获取审计统计"""
    result = await get_room_stats(room_id)
    return result


# ================= 启动入口 =================

if __name__ == "__main__":
    uvicorn.run(
        "pcm_lsjs:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
