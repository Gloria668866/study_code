"""5层企业级 NLU 引擎。

Layer 1: 规则预筛（问候/超纲快速通道，不调 LLM）
Layer 2: LLM 意图分类（单一任务，few-shot 向量检索增强）
Layer 3: 实体抽取（独立 LLM 调用，单一任务）
Layer 4: 槽位完整性检查（规则，来自 YAML）
Layer 5: 业务规则覆盖（来自 YAML，业务同学可维护）
"""
import threading
from pathlib import Path

import yaml

from .config import NLU_CONFIG_PATH

_cfg_lock = threading.Lock()
_cfg = None


def _load_config() -> dict:
    global _cfg
    with _cfg_lock:
        if _cfg is not None:
            return _cfg
        path = Path(NLU_CONFIG_PATH)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _cfg = raw.get("nlu", {})
        return _cfg


def get_cfg() -> dict:
    if _cfg is None:
        _load_config()
    return _cfg


# ── Layer 1: 规则预筛 ──────────────────────────────────────────────────────────

def layer1_prefilter(question: str, last_assistant: str = "") -> dict | None:
    """规则命中返回意图 dict，未命中返回 None。
    返回 {"force_rag": True} 表示上轮无数据，Layer 2 需覆盖 sql/hybrid。"""
    cfg = get_cfg()
    greetings = cfg.get("greeting_prefixes", [])
    q = question.strip()
    if len(q) <= 8 and any(q.startswith(g) for g in greetings):
        return {"intent": "chat", "confidence": 1.0, "source": "layer1_greeting"}

    no_data_signals = cfg.get("no_data_signals", [])
    if last_assistant and any(sig in last_assistant for sig in no_data_signals):
        return {"force_rag": True}

    return None


# ── Layer 4: 槽位完整性检查 ───────────────────────────────────────────────────

def layer4_check_completeness(intent: str, entities: dict, question: str) -> tuple[bool, list]:
    """仅检查 sql/hybrid；其他意图直接返回 (True, [])。"""
    if intent not in ("sql", "hybrid"):
        return True, []

    cfg = get_cfg()
    slot_rules = cfg.get("slot_rules", {})
    requires = [k for k in slot_rules.get("sql_requires_one_of", []) if k != "explicit_metric"]
    metric_kw = slot_rules.get("explicit_metric_keywords", [])

    has_entity = any(entities.get(k) for k in requires)
    has_metric_kw = any(kw in question for kw in metric_kw)

    if has_entity or has_metric_kw:
        return True, []
    return False, ["请补充：品牌（如比亚迪）、车系（如SU7）、时间（如2025年）或具体指标（如Top10）"]


# ── Layer 5: 业务规则覆盖 ─────────────────────────────────────────────────────

def layer5_business_rules(intent: str, question: str) -> str:
    """按配置规则覆盖意图；无命中则原样返回。"""
    cfg = get_cfg()
    for rule in cfg.get("business_rules", []):
        if any(kw in question for kw in rule.get("if_contains", [])):
            return rule["force_intent"]
    return intent


# ── 置信度门控 ────────────────────────────────────────────────────────────────

def _confidence_gate(cls: dict) -> bool:
    """置信度不足或 top1/top2 差距太小 → 返回 False（触发 clarify）。"""
    cfg = get_cfg()
    threshold = float(cfg.get("confidence_threshold", 0.70))
    gap_min = float(cfg.get("ambiguity_gap", 0.15))

    if cls.get("confidence", 0) < threshold:
        return False
    top2 = cls.get("top2_confidence", 0.0)
    if (cls["confidence"] - top2) < gap_min:
        return False
    return True
