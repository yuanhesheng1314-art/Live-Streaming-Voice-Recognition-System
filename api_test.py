# -*- coding: utf-8 -*-
"""
REST API 管理工具测试脚本
用于测试违禁词上传、日志查询、统计等接口
"""
import httpx
import json
import sys

BASE_URL = "http://127.0.0.1:8001"


def print_json(data, indent=2):
    print(json.dumps(data, ensure_ascii=False, indent=indent))


def health_check():
    """健康检查"""
    resp = httpx.get(f"{BASE_URL}/api/health")
    print(f"\n{'='*40}")
    print(" 健康检查")
    print(f"{'='*40}")
    print_json(resp.json())


def upload_words():
    """上传违禁词表"""
    words = [
        {"word": "国家级", "level": "high", "category": "虚假宣传", "note": ""},
        {"word": "最高级", "level": "high", "category": "虚假宣传", "note": ""},
        {"word": "最佳", "level": "high", "category": "虚假宣传", "note": ""},
        {"word": "第一", "level": "high", "category": "虚假宣传", "note": ""},
        {"word": "全网最低价", "level": "high", "category": "虚假宣传", "note": ""},
        {"word": "加微信", "level": "high", "category": "私下交易", "note": ""},
        {"word": "返利", "level": "high", "category": "私下交易", "note": ""},
        {"word": "根治", "level": "high", "category": "医疗健康", "note": ""},
        {"word": "包治百病", "level": "high", "category": "医疗健康", "note": ""},
        {"word": "今天最后一天", "level": "medium", "category": "逼单话术", "note": ""},
        {"word": "马上下架", "level": "medium", "category": "逼单话术", "note": ""},
    ]
    resp = httpx.post(
        f"{BASE_URL}/api/words/upload",
        json={"words": words},
        timeout=30
    )
    print(f"\n{'='*40}")
    print(" 上传违禁词表")
    print(f"{'='*40}")
    print_json(resp.json())


def reload_words():
    """手动重载词库"""
    resp = httpx.post(f"{BASE_URL}/api/words/reload")
    print(f"\n{'='*40}")
    print(" 手动重载词库")
    print(f"{'='*40}")
    print_json(resp.json())


def get_violations():
    """查询违规记录"""
    resp = httpx.get(f"{BASE_URL}/api/logs/violations", params={"limit": 10})
    print(f"\n{'='*40}")
    print(" 最近违规记录")
    print(f"{'='*40}")
    print_json(resp.json())


def get_stats():
    """获取审计统计"""
    resp = httpx.get(f"{BASE_URL}/api/stats")
    print(f"\n{'='*40}")
    print(" 审计统计")
    print(f"{'='*40}")
    print_json(resp.json())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 运行所有测试
        health_check()
        upload_words()
        reload_words()
        get_violations()
        get_stats()
    else:
        cmd = sys.argv[1]
        commands = {
            "health": health_check,
            "upload": upload_words,
            "reload": reload_words,
            "violations": get_violations,
            "stats": get_stats,
        }
        if cmd in commands:
            commands[cmd]()
        else:
            print(f"未知命令: {cmd}")
            print(f"可用命令: {', '.join(commands.keys())}")
