# -*- coding: utf-8 -*-
"""
谐音/变体/绕过检测模块 - A100优化版
检测主播用同音字、拼音、拆字、特殊字符等方式绕过违禁词检测
优化：缓存、批量遍历、减少重复字符串操作，提升检测速度
"""
import re
import json
import functools
from pathlib import Path
from typing import List, Dict, Set
import time
import asyncio

# ============ 常见谐音字映射（直播高频绕过场景 - 全面扩展版） ============
HOMOPHONE_MAP = {
    # 钱类（全面覆盖）
    "钱": ["前", "浅", "欠", "茜", "千", "牵", "签", "谦", "乾", "潜", "佥", "荨", "qian", "￥", "¥", "rmb", "RMB", "软妹币", "米", "钻", "金币"],
    "微信": ["威信", "薇心", "V信", "vx", "VX", "v信", "微x", "维信", "w x", "wei xin", "weixin", "绿泡泡", "绿信", "vx号"],
    "支付宝": ["吱付宝", "支F宝", "ZFB", "zfb", "支付宝F", "支f宝", "zhi付宝", "芝付宝", "知付宝", "支f b", "支f.b"],
    "人民币": ["人民B", "RMB", "rmb", "软妹币", "人名币", "人冥币", "renminbi"],
    "转账": ["转帐", "转zhang", "zhuang帐", "转zhang", "转帐", "zuan zhang", "zhuanhang"],
    "付款": ["付宽", "fukuan", "付kuan", "fu宽", "付钱", "fu qian"],
    "收款": ["收宽", "shoukuan", "收kuan", "shou宽", "收钱", "shou qian"],

    # 交易类（全面覆盖）
    "购买": ["勾买", "go买", "购mai", "goumai", "gou买", "购mai", "勾mai", "购mai"],
    "下单": ["下dan", "下丹", "夏单", "下蛋", "下单子", "xiadan", "下d", "xia dan"],
    "价格": ["价ge", "架格", "JG", "jg", "jia ge", "价位", "价ge", "架格", "价钱"],
    "多少钱": ["多少米", "多少Q", "多少qian", "多少￥", "啥价位", "多钱", "多少钱", "duo shao qian", "多少钱", "多少rmb", "多少软妹币"],
    "便宜": ["便亿", "pianyi", "bian yi", "便意", "便宜", "便yi", "低价"],
    "优惠": ["优惠", "youhui", "you hui", "优惠hui", "优hui"],
    "打折": ["da zhe", "da折", "打zhe", "折价", "折扣"],

    # 引流类（全面覆盖）
    "关注": ["关zhu", "关猪", "guan注", "guanzhu", "guan zhu", "关住", "gz", "GZ", "关主", "关著", "加关注", "点关注", "关注我"],
    "私信": ["思信", "丝信", "si信", "sixin", "si xin", "私聊", "si聊", "私聊", "s聊", "私心", "私性", "私信我", "私信"],
    "加微信": ["加v", "➕V", "jia微信", "加V信", "加vx", "加v信", "加薇心", "加威信", "加绿泡泡", "加w", "加wei xin", "加绿信"],
    "群": ["裙", "qun", "羣", "q un", "q群", "裙号", "群号", "进群", "进裙", "加群", "拉群"],
    "粉丝": ["粉si", "fen丝", "fen si", "fensi", "粉死", "粉s", "粉丝群"],
    "直播间": ["直播间", "直播jian", "zhi播间", "zhibo间", "zb间", "直播间"],

    # 极限词类（全面覆盖）
    "最好": ["zui好", "蕞好", "醉好", "zui hao", "zuihao", "最hao", "z好", "最 Hao"],
    "第一": ["DI一", "No1", "NO.1", "no.1", "No 1", "di一", "第yi", "di yi", "第一名", "第一名", "榜首", "冠军", "d 1"],
    "全网": ["全wang", "全网wang", "quan wang", "quan网", "全w", "qw", "全网第一", "全网最低"],
    "绝对": ["jue对", "绝dui", "决对", "jue dui", "juedui", "绝 d", "jue对", "绝dui", "ju对", "绝怼"],
    "唯一": ["唯yi", "weiyi", "wei yi", "唯y", "w y", "唯伊", "唯一", "独一", "独yi"],
    "独家": ["独jia", "dujia", "du 家", "独 家", "独j", "d j", "独家", "独jia"],
    "顶级": ["顶ji", "dingji", "ding 级", "顶 级", "顶j", "d j", "顶级", "顶ji", "ding级", "顶g级"],
    "极致": ["极zhi", "jizhi", "ji 致", "极 致", "极z", "j z", "极致", "极zhi", "ji致"],
    "国家级": ["国家ji", "guojia级", "国ji级", "国家级", "g j j", "guo家级", "国家级"],

    # 功效类（全面覆盖）
    "治疗": ["治liao", "治聊", "治廖", "zhi liao", "zhiliao", "治 z", "zhi疗", "治liao", "zhi疗", "疗治"],
    "治愈": ["治yu", "治俞", "zhi愈", "zhi yu", "zhiyu", "治 y", "z y", "治愈", "治yu", "zhi愈"],
    "疗效": ["xiao果", "效guo", "xiaoguo", "xiao 果", "效 果", "效g", "x g", "疗效", "效guo", "xiao果"],
    "效果": ["效guo", "xiao果", "效果", "xiao guo", "xiao果", "效g", "x g", "效果"],
    "根治": ["gen治", "根zhi", "根治", "gen 治", "根 治", "根z", "g z", "根治", "根zhi"],
    "减肥": ["jian肥", "减fei", "jian fei", "jianfei", "减 f", "j f", "减肥", "减fei", "jian肥", "瘦身", "shou身"],
    "瘦身": ["shou身", "瘦shen", "瘦身", "shou shen", "shoushen", "瘦 s", "s s"],

    # 其他常见（全面覆盖）
    "赚钱": ["zhuan钱", "赚qian", "砖钱", "zhuan qian", "zhuanqian", "赚 q", "z q", "赚钱", "赚qian", "zhuan钱", "挣钱", "zheng钱"],
    "返利": ["fan利", "返li", "fanli", "fan 利", "返 利", "返l", "f l", "返利", "返li", "fan利"],
    "代购": ["代gou", "代go", "daigou", "dai gou", "代 g", "d g", "代购", "代gou", "dai购"],
    "刷单": ["刷dan", "唰单", "shua单", "shuadan", "刷 d", "s d", "刷单", "刷dan", "唰dan"],
    "兼职": ["jian职", "兼zhi", "jianzhi", "jian 职zhi", "兼 职", "兼z", "j z", "兼职", "兼zhi"],

    # 平台类
    "淘宝": ["淘bao", "taobao", "tb", "TB", "淘b", "t b", "淘宝", "淘宝网"],
    "京东": ["京dong", "jd", "JD", "京d", "j d", "京东", "京东网"],
    "拼多多": ["拼多duo", "pdd", "PDD", "拼dd", "拼 d d", "拼多多"],
    "抖音": ["抖yin", "douyin", "dy", "DY", "抖y", "d y", "抖音", "抖yin"],
    "快手": ["快shou", "kuai手", "ks", "KS", "快s", "k s", "快手"],

    # 敏感品类
    "涉黄": ["涉huang", "she黄", "涉h", "s h", "黄色", "huang色"],
    "博彩": ["博cai", "bo彩", "bocai", "博 c", "b c", "博彩", "彩票", "cai票"],
    "彩票": ["彩piao", "cai票", "彩票", "cai piao", "彩 p", "c p"],
    "赌博": ["赌bo", "du博", "du博", "du bo", "dubo", "赌 b", "d b"],

    # 常见组合词
    "包治百病": ["包zhi百病", "包治bai病", "包zhi bai bing", "包治b病", "bao zhi bai bing"],
    "药到病除": ["药dao病除", "药到bing除", "药 d b c", "yao dao bing chu"],
    "百分百": ["百分bai", "百fen百", "bai分bai", "百分b", "100%", "百分百"],
    "稳赚不赔": ["稳zhuan不赔", "稳赚bu赔", "wen zhan bu pei", "稳z b p"],
    "保本": ["保ben", "bao本", "保 b", "b b", "保本", "bao ben"],

    # 联系方式类
    "电话": ["电hua", "dian话", "dh", "DH", "电h", "d h", "电话", "手机", "shou ji"],
    "手机": ["手ji", "shou机", "sj", "SJ", "手j", "s j", "手机", "号码", "haoma"],
    "号码": ["号ma", "hao码", "hm", "HM", "号m", "h m", "号码", "手机号"],
    "QQ": ["qq", "扣扣", "kou kou", "k k", "扣k", "q扣"],
    "邮箱": ["邮xiang", "you箱", "yx", "YX", "邮x", "y x", "邮箱", "email", "e mail"],
}

# 特殊字符（用于插入到违禁词中间绕过检测）- 扩展版
# 末段原为不可见 Unicode 区间，若在 r"..." 中间物理换行会触发 SyntaxError（未闭合字符串），
# 此处用 \u 转义单行书写；\u2028-\u206f 覆盖行/段分隔、Bidi 控制、窄空格 U+202F、数学空格 U+205F、词连接符 U+2060–U+206F 等。
SPECIAL_CHARS_PATTERN = re.compile(
    r"[^一-龥〇 -⁯a-zA-Z0-9]+"
    + r"|[\s　﻿ ]+"
    + r"|[\.\-\_\|/\\\*\&\%\$\#\@\!\~\`\+\=\^\(\)\[\]\{\}\;\:\'\"\<\>\?]+"
    # 原字面量「空格-\x1f」在字符类里起点码点大于终点，触发 re.error: bad character range
    + r"|[\x00-\x1f\x7f-\x9f]+"
    + r"|[\u200b-\u200f\u2028-\u206f\ufeff]+"
)

# 数字变体映射
NUMBER_VARIANT_MAP = {
    "0": ["零", "〇", "o", "O", "０", "𝟎", "𝟘"],
    "1": ["一", "壹", "i", "I", "l", "L", "１", "𝟏", "𝟙"],
    "2": ["二", "贰", "ii", "II", "２", "𝟐", "𝟚"],
    "3": ["三", "叁", "iii", "III", "３", "𝟑", "𝟛"],
    "4": ["四", "肆", "iv", "IV", "４", "𝟒", "𝟜"],
    "5": ["五", "伍", "v", "V", "５", "𝟓", "𝟝"],
    "6": ["六", "陆", "vi", "VI", "６", "𝟔", "𝟞"],
    "7": ["七", "柒", "vii", "VII", "７", "𝟕", "𝟟"],
    "8": ["八", "捌", "viii", "VIII", "８", "𝟖", "𝟠"],
    "9": ["九", "玖", "ix", "IX", "９", "𝟗", "𝟡"],
}

# 构建数字反向映射
REVERSE_NUMBER_MAP = {}
for digit, variants in NUMBER_VARIANT_MAP.items():
    for v in variants:
        REVERSE_NUMBER_MAP[v] = digit


def _build_reverse_homophone() -> dict[str, str]:
    """构建反向谐音映射: 谐音字 -> 原字（lru_cache 避免重复构建）"""
    reverse = {}
    for original, variants in HOMOPHONE_MAP.items():
        for v in variants:
            reverse[v] = original
    return reverse


REVERSE_HOMOPHONE = _build_reverse_homophone()

# 将 REVERSE_HOMOPHONE 转为 frozenset 用于快速 O(1) 成员检查，避免每句遍历所有键
_REVERSE_HOMO_KEYS = frozenset(REVERSE_HOMOPHONE.keys())
# 预编译：两字以上变体的正则（用 | 连接减少多次 re.search）
_DIRECT_HOMO_PATTERN = re.compile(
    r"(?:" + "|".join(sorted((k for k in _REVERSE_HOMO_KEYS if len(k) >= 2), key=len, reverse=True)) + r")"
)


# A100优化：添加文本缓存
@functools.lru_cache(maxsize=2048)
def normalize_text_cached(text: str) -> str:
    """文本归一化：去除空格、标点、特殊符号（带缓存）"""
    return re.sub(r"[\s\-\_\|,，。！？!?:：;；·…]+", "", text or "")


@functools.lru_cache(maxsize=1024)
def strip_special_chars_cached(text: str) -> str:
    """移除特殊字符，还原被插入符号分隔的违禁词（带缓存）"""
    return SPECIAL_CHARS_PATTERN.sub("", text)


@functools.lru_cache(maxsize=1024)
def normalize_numbers_cached(text: str) -> str:
    """将数字变体统一转换为标准数字（用于检测数字绕过）（带缓存）"""
    result = text
    for variant, digit in REVERSE_NUMBER_MAP.items():
        if variant in result:
            result = result.replace(variant, digit)
    return result


def restore_homophones(text: str) -> str:
    """将谐音字替换回原字，用于二次匹配（多层检测）"""
    result = text
    # 先处理数字变体
    result = normalize_numbers_cached(result)
    # 再处理谐音字
    for variant, original in REVERSE_HOMOPHONE.items():
        if variant in result:
            result = result.replace(variant, original)
    return result


@functools.lru_cache(maxsize=2048)
def generate_variants_cached(word: str) -> list[str]:
    """为一个违禁词生成常见的谐音/变体形式（全面扩展 + lru_cache 缓存）"""
    variants = {word}
    chars = list(word)

    # 逐字替换谐音
    for i, ch in enumerate(chars):
        if ch in HOMOPHONE_MAP:
            for alt in HOMOPHONE_MAP[ch][:8]:  # 每个字最多8种变体，控制组合爆炸
                alt_chars = chars[:]
                alt_chars[i] = alt
                variants.add("".join(alt_chars))

    # 数字变体
    for i, ch in enumerate(chars):
        if ch in NUMBER_VARIANT_MAP:
            for alt in NUMBER_VARIANT_MAP[ch][:5]:
                alt_chars = chars[:]
                alt_chars[i] = alt
                variants.add("".join(alt_chars))

    # 拼音缩写变体（仅适用于英文拼音输入）
    if len(word) >= 2:
        pinyin_abbrev = ""
        for ch in word:
            if '一' <= ch <= '龥':  # 中文
                pinyin_abbrev += ch
            elif ch.isalpha():
                pinyin_abbrev += ch
        if pinyin_abbrev and pinyin_abbrev != word:
            variants.add(pinyin_abbrev.upper())

    # 首字母缩写变体（针对常见组合词）
    if len(word) >= 3:
        abbrev = ""
        for ch in word:
            if '一' <= ch <= '龥':
                abbrev += ch[0] if ch else ""
            elif ch.isalpha():
                abbrev += ch.upper()
        if abbrev and abbrev != word and len(abbrev) >= 2:
            variants.add(abbrev)

    return list(variants)


class VariantDetector:
    """变体检测器（增强版 A100优化）"""

    def __init__(self):
        self.variant_map: dict[str, str] = {}  # 变体 -> 原词
        self._last_words: list[str] = []  # 上次的词库
        self._last_build_time: float = 0  # 上次构建时间

    def load_from_words(self, words: list[str]):
        """从违禁词列表生成变体映射（仅在词库变化时重建）"""
        if self._last_words == words and time.time() - self._last_build_time < 60:
            # 词库未变化且缓存未过期，跳过重建
            return

        self.variant_map.clear()
        for word in words:
            for variant in generate_variants_cached(word):
                if variant != word:
                    self.variant_map[variant] = word

        self._last_words = words.copy()
        self._last_build_time = time.time()

    def detect(self, text: str) -> list[dict]:
        """
        检测变体绕过（多层检测，A100 优化版）
        - 单次迭代覆盖多个方法（减少遍历 variant_map 次数）
        - 使用缓存优化
        - 批量处理减少正则调用
        返回: [{"original": "原词", "variant": "变体形式", "method": "检测方法"}]
        """
        results: list[dict] = []

        # 0. 特殊字符插入检测：清理文本后匹配（使用缓存）
        stripped = strip_special_chars_cached(text)

        # 1. 谐音字检测（原始文本 + 清理后文本合并为一次遍历 variant_map）
        restored = restore_homophones(text)

        # 2. 清理后再做谐音检测（使用缓存）
        if stripped != text:
            restored_stripped = restore_homophones(stripped)

        # 3. 数字变体检测（使用缓存）
        normalized = normalize_numbers_cached(text)

        # 统一遍历 variant_map（原词 + 所有变体映射），避免循环4次
        seen_variants: set[str] = set()
        methods_seen: set[tuple[str, str]] = set()

        for variant, original in self.variant_map.items():
            matched_method = None
            if stripped != text and variant in stripped:
                matched_method = "special_char_bypass"
            elif restored != text and variant in restored:
                matched_method = "homophone"
            elif stripped != text and variant in restored_stripped:
                matched_method = "combined_bypass"
            elif normalized != text and variant in normalized:
                matched_method = "number_variant"

            if matched_method is not None and variant not in seen_variants:
                seen_variants.add(variant)
                results.append({
                    "original": original,
                    "variant": variant,
                    "method": matched_method
                })

        # 4. 通用谐音替换（使用预编译正则单次匹配，替代逐字 in 检查）
        for m in _DIRECT_HOMO_PATTERN.finditer(text):
            homo_match = m.group(0)
            original = REVERSE_HOMOPHONE.get(homo_match)
            if original:
                key = (original, homo_match)
                if key not in methods_seen:
                    methods_seen.add(key)
                    results.append({
                        "original": original,
                        "variant": homo_match,
                        "method": "direct_homophone"
                    })

        # 去重（保留置信度更高的检测方法）
        dedup: dict[tuple[str, str], dict] = {}
        method_priority = {"direct_homophone": 1, "homophone": 2, "combined_bypass": 3, "special_char_bypass": 4, "number_variant": 5}
        for r in results:
            key = (r["original"], r.get("variant", ""))
            existing = dedup.get(key)
            if existing is None or method_priority.get(r.get("method", ""), 0) > method_priority.get(existing.get("method"), 0):
                dedup[key] = r

        return list(dedup.values())


# A100优化：批量检测接口
async def batch_detect_variants(detector: VariantDetector, texts: List[str]) -> List[List[dict]]:
    """批量变体检测，利用并发提升效率"""
    loop = asyncio.get_event_loop()

    # 创建任务列表
    tasks = [loop.run_in_executor(None, detector.detect, text) for text in texts]

    # 并发执行
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"变体检测异常: {result}")
            processed_results.append([])
        else:
            processed_results.append(result)

    return processed_results


# A100优化：热更新接口
async def hot_reload_detector(detector: VariantDetector, words: List[str]):
    """热更新词库（不阻塞主线程）"""
    try:
        # 在线程池中执行耗时操作
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, detector.load_from_words, words)
        print("变体检测器热更新完成")
    except Exception as e:
        print(f"变体检测器热更新失败: {e}")