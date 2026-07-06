"""5层企业级 NLU 引擎。

Layer 1: 规则预筛（问候/超纲快速通道，不调 LLM）
Layer 2: LLM 意图分类（单一任务，few-shot 向量检索增强）
Layer 3: 实体抽取（独立 LLM 调用，单一任务）
Layer 4: 槽位完整性检查（规则，来自 YAML）
Layer 5: 业务规则覆盖（来自 YAML，业务同学可维护）
"""
import json
import threading
from pathlib import Path

import numpy as np
import yaml

from .config import NLU_CONFIG_PATH
from .llm import chat
from .rag.embed import embed_query, embed_passages

_cfg_lock = threading.Lock()
_cfg = None

_few_shot_lock = threading.Lock()
_few_shot_embeddings = None   # np.ndarray shape (N, embed_dim)
_few_shot_examples = None     # list[dict]


def _load_config() -> dict:
    global _cfg, _few_shot_embeddings, _few_shot_examples
    with _cfg_lock:
        if _cfg is not None:
            return _cfg
        path = Path(NLU_CONFIG_PATH)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _cfg = raw.get("nlu", {})
        # Pre-embed few-shot examples into in-memory numpy array
        examples = _cfg.get("few_shot_examples") or []
        if examples:
            texts = [e["question"] for e in examples]
            vecs = embed_passages(texts)   # returns [] if model unavailable
            if vecs:
                _few_shot_embeddings = np.array(vecs, dtype=np.float32)
                _few_shot_examples = examples
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


# ── Few-shot 向量检索 ──────────────────────────────────────────────────────────

def _retrieve_few_shot(question: str, top_k: int = 3) -> list:
    """余弦相似度检索 in-memory few-shot；模型不可用或未加载返回 []。"""
    if _few_shot_embeddings is None or _few_shot_examples is None:
        return []
    qvec = embed_query(question)
    if qvec is None:
        return []
    q = np.array(qvec, dtype=np.float32)
    sims = _few_shot_embeddings @ q          # 已归一化，点积=余弦相似度
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [_few_shot_examples[i] for i in top_idx]


def _build_few_shot_block(examples: list) -> str:
    if not examples:
        return ""
    lines = ["参考示例："]
    for ex in examples:
        lines.append(f'  问题: "{ex["question"]}" → intent: {ex["intent"]}')
    return "\n".join(lines) + "\n\n"


# ── Layer 2: LLM 意图分类（单一任务）─────────────────────────────────────────

_CLASSIFY_SYS = """\
你是新能源汽车市场情报系统的意图分类器。只做分类，不做实体提取。返回严格 JSON（无其他文字）。

JSON 格式：
{"intent":"sql|rag|hybrid|chat|clarify","confidence":0.95,"top2_intent":"rag","top2_confidence":0.08}

意图定义：
- sql: 需要查结构化数据库（销量/价格/排名/趋势/对比/统计数字）
- rag: 需要从文档/研报/政策/口碑检索（为什么/解读/分析/评价/预测/观点）
- hybrid: 既需要数据统计又需要文档解读
- chat: 问候/闲聊/完全无关新能源汽车市场
- clarify: 问题模糊/缺少必要实体/无法确定查询口径
"""


def layer2_classify(question: str, context_block: str = "", force_rag: bool = False) -> dict:
    """单任务 LLM 分类 + few-shot 增强。FAIL-SAFE：异常返回 clarify。"""
    few_shot = _retrieve_few_shot(question)
    prompt = _build_few_shot_block(few_shot)
    if context_block:
        prompt += context_block + "\n\n"
    prompt += question

    try:
        raw = chat([
            {"role": "system", "content": _CLASSIFY_SYS},
            {"role": "user", "content": prompt},
        ], temperature=0.0)
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0 or e <= s:
            raise ValueError("no JSON in response")
        r = json.loads(raw[s:e])
        intent = r.get("intent", "clarify")
        if intent not in ("sql", "rag", "hybrid", "chat", "clarify"):
            intent = "clarify"
        if force_rag and intent in ("sql", "hybrid"):
            intent = "rag"
        return {
            "intent": intent,
            "confidence": min(1.0, max(0.0, float(r.get("confidence", 0.8)))),
            "top2_intent": r.get("top2_intent", ""),
            "top2_confidence": float(r.get("top2_confidence", 0.0)),
            "source": "layer2_llm",
        }
    except Exception:
        return {"intent": "clarify", "confidence": 0.5,
                "top2_intent": "", "top2_confidence": 0.0,
                "source": "layer2_fallback"}
