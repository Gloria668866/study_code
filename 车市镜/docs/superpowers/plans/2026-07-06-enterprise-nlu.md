# Enterprise NLU Reform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic single-LLM-call intent router in `app/graph.py` with a 5-layer enterprise NLU engine that separates concerns, adds confidence gating, and loads few-shot examples + business rules from a YAML config file.

**Architecture:** A new `app/nlu.py` module owns all 5 layers (rule pre-filter → LLM classify → entity extract → slot check → business rules). The existing `intent_router` node in `app/graph.py` becomes a thin wrapper that calls `nlu.classify()` and handles cross-turn entity memory. Few-shot examples are embedded at startup into an in-memory numpy array for fast cosine-similarity retrieval — no new database needed.

**Tech Stack:** LangGraph (existing), sentence-transformers BGE-large-zh (existing), numpy (existing via torch), PyYAML, existing `app/llm.py` chat interface.

## Global Constraints

- Python 3.11+, FastAPI/LangGraph existing stack
- Do NOT change `AgentState` schema or any node other than `intent_router`
- All LLM calls use existing `app/llm.py chat()` — do not add new LLM clients
- YAML config must be loaded once at startup (thread-safe lazy init, double-checked locking)
- Every layer except Layer 2 and Layer 3 must be testable without mocking LLM
- `app/graph.py` must still export `_classify_intent`, `_GREETING_PREFIXES`, `_NO_DATA_SIGNALS` for the existing `test_graph.py` (backwards compat shims)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| CREATE | `config/nlu.yaml` | Greeting prefixes, no-data signals, few-shot examples, slot rules, business rules, thresholds |
| CREATE | `app/nlu.py` | 5-layer NLU engine + few-shot in-memory retrieval |
| MODIFY | `app/config.py` | Add `NLU_CONFIG_PATH` env var |
| MODIFY | `app/graph.py` | Replace `intent_router` body to call `nlu.classify()`; add backwards-compat shims |
| CREATE | `tests/test_nlu.py` | Unit tests for all 5 layers + classify() integration |

---

## Task 1: Config file + app/config.py change

**Files:**
- Create: `config/nlu.yaml`
- Modify: `app/config.py` (add 1 line)

**Interfaces:**
- Produces: `config/nlu.yaml` loaded by `app/nlu.py` via `NLU_CONFIG_PATH`

- [ ] **Step 1: Create `config/nlu.yaml`**

```yaml
nlu:
  confidence_threshold: 0.70   # below this → clarify
  ambiguity_gap: 0.15          # if top1_conf - top2_conf < this → clarify

  greeting_prefixes:
    - "你好"
    - "您好"
    - "早安"
    - "晚安"
    - "早上好"
    - "再见"
    - "拜拜"

  no_data_signals:
    - "no data"
    - "not found"
    - "no results"
    - "no records"
    - "未查询到"
    - "没有找到"
    - "0条结果"
    - "不在覆盖范围"
    - "不在数据库"
    - "未在知识库中检索到"
    - "未检索到"

  few_shot_examples:
    - question: "比亚迪2025年各月销量是多少"
      intent: sql
    - question: "特斯拉Model Y今年卖了多少台"
      intent: sql
    - question: "2025年纯电Top10排名"
      intent: sql
    - question: "理想L9和小米SU7谁卖得多"
      intent: sql
    - question: "小米SU7口碑怎么样"
      intent: rag
    - question: "最近的新能源购车补贴政策有哪些"
      intent: rag
    - question: "比亚迪为什么能卖那么好"
      intent: rag
    - question: "问界M9的用户评价如何"
      intent: rag
    - question: "理想L9销量如何，为什么这么好卖"
      intent: hybrid
    - question: "2025年纯电销量Top5，分析各品牌优势"
      intent: hybrid
    - question: "你好"
      intent: chat
    - question: "谢谢"
      intent: chat
    - question: "销量排名"
      intent: clarify
    - question: "那辆车"
      intent: clarify
    - question: "最近的数据"
      intent: clarify

  slot_rules:
    sql_requires_one_of:
      - brands
      - models
      - time
      - energy_types
      - explicit_metric
    explicit_metric_keywords:
      - "top"
      - "Top"
      - "排名"
      - "对比"
      - "比较"
      - "最多"
      - "第一"
      - "销量榜"
      - "增长"
      - "下滑"

  business_rules:
    - if_contains: ["政策", "补贴", "购置税"]
      force_intent: rag
    - if_contains: ["法规", "标准", "规定", "条例"]
      force_intent: rag
    - if_contains: ["为什么", "原因", "分析", "解读", "预测", "展望"]
      force_intent: rag
```

- [ ] **Step 2: Add `NLU_CONFIG_PATH` to `app/config.py`**

Open `app/config.py` and add this line after the existing `RERANK_SCORE_MIN` line:

```python
NLU_CONFIG_PATH = os.getenv("NLU_CONFIG_PATH", "config/nlu.yaml")
```

- [ ] **Step 3: Commit**

```bash
git add config/nlu.yaml app/config.py
git commit -m "feat(nlu): add nlu.yaml config and NLU_CONFIG_PATH env var"
```

---

## Task 2: `app/nlu.py` — Layers 1, 4, 5 + confidence gate (no LLM)

**Files:**
- Create: `app/nlu.py`
- Create: `tests/test_nlu.py` (layers 1, 4, 5, confidence gate)

**Interfaces:**
- Consumes: `app/config.NLU_CONFIG_PATH`, `config/nlu.yaml`
- Produces:
  - `nlu.get_cfg() -> dict`
  - `nlu.layer1_prefilter(question, last_assistant) -> dict | None`
  - `nlu.layer4_check_completeness(intent, entities, question) -> tuple[bool, list]`
  - `nlu.layer5_business_rules(intent, question) -> str`
  - `nlu._confidence_gate(cls_dict) -> bool`  (cls_dict has keys: confidence, top2_confidence)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nlu.py`:

```python
"""Unit tests for app/nlu.py — layers that don't need LLM."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ── Layer 1 ───────────────────────────────────────────────────────────────────

def test_layer1_greeting_short():
    from app.nlu import layer1_prefilter
    result = layer1_prefilter("你好", "")
    assert result is not None
    assert result["intent"] == "chat"
    assert result["source"] == "layer1_greeting"


def test_layer1_greeting_long_not_triggered():
    from app.nlu import layer1_prefilter
    result = layer1_prefilter("你好，比亚迪今年卖了多少", "")
    assert result is None


def test_layer1_no_data_signal_sets_force_rag():
    from app.nlu import layer1_prefilter
    result = layer1_prefilter("进一步分析", "未查询到相关数据，请换个问法")
    assert result is not None
    assert result.get("force_rag") is True


def test_layer1_no_signal_returns_none():
    from app.nlu import layer1_prefilter
    result = layer1_prefilter("比亚迪销量", "比亚迪2025年销量为120万辆")
    assert result is None


# ── Layer 4 ───────────────────────────────────────────────────────────────────

def test_layer4_sql_with_brand_is_complete():
    from app.nlu import layer4_check_completeness
    ok, slots = layer4_check_completeness("sql", {"brands": ["比亚迪"]}, "比亚迪销量")
    assert ok is True
    assert slots == []


def test_layer4_sql_with_metric_keyword_is_complete():
    from app.nlu import layer4_check_completeness
    ok, slots = layer4_check_completeness("sql", {}, "2025年纯电Top10")
    assert ok is True


def test_layer4_sql_no_entity_no_keyword_incomplete():
    from app.nlu import layer4_check_completeness
    ok, slots = layer4_check_completeness("sql", {}, "销量排名")
    assert ok is False
    assert len(slots) > 0


def test_layer4_rag_always_complete():
    from app.nlu import layer4_check_completeness
    ok, slots = layer4_check_completeness("rag", {}, "为什么")
    assert ok is True


def test_layer4_chat_always_complete():
    from app.nlu import layer4_check_completeness
    ok, slots = layer4_check_completeness("chat", {}, "你好")
    assert ok is True


# ── Layer 5 ───────────────────────────────────────────────────────────────────

def test_layer5_policy_keyword_forces_rag():
    from app.nlu import layer5_business_rules
    assert layer5_business_rules("sql", "最新的新能源购置税政策") == "rag"


def test_layer5_subsidy_forces_rag():
    from app.nlu import layer5_business_rules
    assert layer5_business_rules("hybrid", "比亚迪能享受哪些补贴") == "rag"


def test_layer5_no_rule_match_unchanged():
    from app.nlu import layer5_business_rules
    assert layer5_business_rules("sql", "比亚迪2025年销量") == "sql"


# ── Confidence gate ───────────────────────────────────────────────────────────

def test_confidence_gate_passes_high():
    from app.nlu import _confidence_gate
    assert _confidence_gate({"confidence": 0.95, "top2_confidence": 0.10}) is True


def test_confidence_gate_fails_low_confidence():
    from app.nlu import _confidence_gate
    assert _confidence_gate({"confidence": 0.60, "top2_confidence": 0.20}) is False


def test_confidence_gate_fails_ambiguous():
    from app.nlu import _confidence_gate
    # gap = 0.75 - 0.65 = 0.10 < threshold 0.15
    assert _confidence_gate({"confidence": 0.75, "top2_confidence": 0.65}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:/Users/GUANGBL/lgb_coding/demo1
.venv/Scripts/pytest tests/test_nlu.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'layer1_prefilter' from 'app.nlu'` (module doesn't exist yet)

- [ ] **Step 3: Create `app/nlu.py` with Layers 1, 4, 5 + confidence gate**

```python
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
    if top2 and (cls["confidence"] - top2) < gap_min:
        return False
    return True
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/pytest tests/test_nlu.py -v -k "layer1 or layer4 or layer5 or confidence_gate"
```

Expected: all 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/nlu.py tests/test_nlu.py config/nlu.yaml app/config.py
git commit -m "feat(nlu): layers 1/4/5 + confidence gate, all tests green"
```

---

## Task 3: `app/nlu.py` — Layer 2 (LLM classify) + few-shot in-memory retrieval

**Files:**
- Modify: `app/nlu.py` (add Layer 2 + few-shot init)
- Modify: `tests/test_nlu.py` (add Layer 2 tests with mocked LLM)

**Interfaces:**
- Consumes: `app/rag/embed.embed_query()`, `app/rag/embed.embed_passages()`, `app/llm.chat()`
- Produces: `nlu.layer2_classify(question, context_block, force_rag) -> dict`
  - keys: `intent`, `confidence`, `top2_intent`, `top2_confidence`, `source`

- [ ] **Step 1: Write failing tests for Layer 2**

Add to `tests/test_nlu.py`:

```python
# ── Layer 2 ───────────────────────────────────────────────────────────────────

def test_layer2_classify_sql(monkeypatch):
    from app import nlu
    monkeypatch.setattr(nlu, "_few_shot_embeddings", None)
    monkeypatch.setattr(nlu, "_few_shot_examples", None)
    with patch("app.nlu.chat", return_value='{"intent":"sql","confidence":0.95,"top2_intent":"hybrid","top2_confidence":0.08}'):
        result = nlu.layer2_classify("比亚迪2025年销量", "")
    assert result["intent"] == "sql"
    assert result["confidence"] == 0.95
    assert result["source"] == "layer2_llm"


def test_layer2_classify_force_rag_overrides_sql(monkeypatch):
    from app import nlu
    monkeypatch.setattr(nlu, "_few_shot_embeddings", None)
    monkeypatch.setattr(nlu, "_few_shot_examples", None)
    with patch("app.nlu.chat", return_value='{"intent":"sql","confidence":0.90,"top2_intent":"rag","top2_confidence":0.10}'):
        result = nlu.layer2_classify("进一步分析", "", force_rag=True)
    assert result["intent"] == "rag"


def test_layer2_classify_llm_error_returns_clarify(monkeypatch):
    from app import nlu
    monkeypatch.setattr(nlu, "_few_shot_embeddings", None)
    monkeypatch.setattr(nlu, "_few_shot_examples", None)
    with patch("app.nlu.chat", side_effect=Exception("timeout")):
        result = nlu.layer2_classify("比亚迪销量", "")
    assert result["intent"] == "clarify"
    assert result["source"] == "layer2_fallback"


def test_layer2_few_shot_retrieval_returns_examples(monkeypatch):
    import numpy as np
    from app import nlu
    fake_emb = np.random.rand(3, 4).astype(np.float32)
    # normalize
    fake_emb = fake_emb / np.linalg.norm(fake_emb, axis=1, keepdims=True)
    fake_examples = [
        {"question": "比亚迪销量", "intent": "sql"},
        {"question": "口碑怎么样", "intent": "rag"},
        {"question": "你好", "intent": "chat"},
    ]
    monkeypatch.setattr(nlu, "_few_shot_embeddings", fake_emb)
    monkeypatch.setattr(nlu, "_few_shot_examples", fake_examples)

    query_vec = np.random.rand(4).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)
    with patch("app.nlu.embed_query", return_value=query_vec.tolist()):
        results = nlu._retrieve_few_shot("销量", top_k=2)
    assert len(results) == 2
    assert all("intent" in r for r in results)
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/Scripts/pytest tests/test_nlu.py -v -k "layer2" 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError` — `layer2_classify` not defined yet.

- [ ] **Step 3: Add Layer 2 + few-shot retrieval to `app/nlu.py`**

Add the following imports at the top of `app/nlu.py` (after existing imports):

```python
import numpy as np

from .llm import chat
from .rag.embed import embed_query, embed_passages
```

Add these module-level variables after `_cfg = None`:

```python
_few_shot_lock = threading.Lock()
_few_shot_embeddings = None   # np.ndarray shape (N, embed_dim)
_few_shot_examples = None     # list[dict]
```

Replace `_load_config` with this version that also pre-embeds few-shot examples:

```python
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
```

Add these new functions after `_confidence_gate`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/pytest tests/test_nlu.py -v -k "layer2"
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/nlu.py tests/test_nlu.py
git commit -m "feat(nlu): layer 2 LLM classify + in-memory few-shot retrieval"
```

---

## Task 4: `app/nlu.py` — Layer 3 (entity extraction) + `classify()` public API

**Files:**
- Modify: `app/nlu.py` (add Layer 3 + public `classify()`)
- Modify: `tests/test_nlu.py` (add Layer 3 + classify() integration tests)

**Interfaces:**
- Produces: `nlu.classify(question, history, context_block, last_assistant) -> dict`
  - keys: `intent`, `confidence`, `entities`, `is_complete`, `missing_slots`, `normalized_question`, `source`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_nlu.py`:

```python
# ── Layer 3 ───────────────────────────────────────────────────────────────────

def test_layer3_extracts_brand(monkeypatch):
    from app import nlu
    with patch("app.nlu.chat", return_value='{"brands":["比亚迪"],"models":[],"time":["2025年"],"metrics":[],"energy_types":[],"normalized_question":"比亚迪2025年销量"}'):
        result = nlu.layer3_extract_entities("比亚迪2025年销量", "")
    assert result["brands"] == ["比亚迪"]
    assert result["time"] == ["2025年"]
    assert result["normalized_question"] == "比亚迪2025年销量"


def test_layer3_llm_error_returns_empty(monkeypatch):
    from app import nlu
    with patch("app.nlu.chat", side_effect=Exception("timeout")):
        result = nlu.layer3_extract_entities("比亚迪销量", "")
    assert result["brands"] == []
    assert result["normalized_question"] == "比亚迪销量"


# ── classify() 完整流程 ────────────────────────────────────────────────────────

def test_classify_full_sql_flow(monkeypatch):
    from app import nlu
    monkeypatch.setattr(nlu, "_few_shot_embeddings", None)
    monkeypatch.setattr(nlu, "_few_shot_examples", None)

    classify_resp = '{"intent":"sql","confidence":0.92,"top2_intent":"hybrid","top2_confidence":0.05}'
    entity_resp = '{"brands":["比亚迪"],"models":[],"time":["2025年"],"metrics":[],"energy_types":[],"normalized_question":"比亚迪2025年销量"}'

    call_count = {"n": 0}
    def mock_chat(msgs, **kw):
        call_count["n"] += 1
        return classify_resp if call_count["n"] == 1 else entity_resp

    with patch("app.nlu.chat", side_effect=mock_chat):
        result = nlu.classify("比亚迪2025年销量")

    assert result["intent"] == "sql"
    assert result["entities"]["brands"] == ["比亚迪"]
    assert result["is_complete"] is True
    assert result["missing_slots"] == []


def test_classify_incomplete_sql_becomes_clarify(monkeypatch):
    from app import nlu
    monkeypatch.setattr(nlu, "_few_shot_embeddings", None)
    monkeypatch.setattr(nlu, "_few_shot_examples", None)

    classify_resp = '{"intent":"sql","confidence":0.88,"top2_intent":"clarify","top2_confidence":0.05}'
    entity_resp = '{"brands":[],"models":[],"time":[],"metrics":[],"energy_types":[],"normalized_question":"销量排名"}'

    call_count = {"n": 0}
    def mock_chat(msgs, **kw):
        call_count["n"] += 1
        return classify_resp if call_count["n"] == 1 else entity_resp

    with patch("app.nlu.chat", side_effect=mock_chat):
        result = nlu.classify("销量排名")

    assert result["intent"] == "clarify"
    assert len(result["missing_slots"]) > 0


def test_classify_policy_question_forced_rag(monkeypatch):
    from app import nlu
    monkeypatch.setattr(nlu, "_few_shot_embeddings", None)
    monkeypatch.setattr(nlu, "_few_shot_examples", None)

    classify_resp = '{"intent":"sql","confidence":0.85,"top2_intent":"rag","top2_confidence":0.10}'
    entity_resp = '{"brands":[],"models":[],"time":[],"metrics":[],"energy_types":[],"normalized_question":"购置税政策"}'

    call_count = {"n": 0}
    def mock_chat(msgs, **kw):
        call_count["n"] += 1
        return classify_resp if call_count["n"] == 1 else entity_resp

    with patch("app.nlu.chat", side_effect=mock_chat):
        result = nlu.classify("最新的购置税政策有哪些")

    assert result["intent"] == "rag"


def test_classify_low_confidence_becomes_clarify(monkeypatch):
    from app import nlu
    monkeypatch.setattr(nlu, "_few_shot_embeddings", None)
    monkeypatch.setattr(nlu, "_few_shot_examples", None)

    classify_resp = '{"intent":"sql","confidence":0.55,"top2_intent":"rag","top2_confidence":0.40}'

    with patch("app.nlu.chat", return_value=classify_resp):
        result = nlu.classify("那个数据")

    assert result["intent"] == "clarify"
    assert result["source"] == "confidence_gate"
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/Scripts/pytest tests/test_nlu.py -v -k "layer3 or classify" 2>&1 | head -20
```

Expected: `AttributeError: module 'app.nlu' has no attribute 'layer3_extract_entities'`

- [ ] **Step 3: Add Layer 3 + `classify()` to `app/nlu.py`**

Add after `layer2_classify`:

```python
# ── Layer 3: 实体抽取（独立 LLM 调用）────────────────────────────────────────

_ENTITY_SYS = """\
你是实体提取器，只做实体提取和规范化，不做意图判断。从新能源汽车问题中提取实体，返回严格 JSON（无其他文字）。

JSON 格式：
{"brands":[],"models":[],"time":[],"metrics":[],"energy_types":[],"normalized_question":"规范化后的问题"}

规范化规则：su7/SU7不变, byd/BYD→比亚迪, 今年→2026年, 去年→2025年, 前年→2024年
"""


def layer3_extract_entities(question: str, context_block: str = "") -> dict:
    """独立 LLM 实体抽取。FAIL-SAFE：异常返回空实体 + 原始问题。"""
    prompt = (context_block + "\n\n" if context_block else "") + question
    empty = {"brands": [], "models": [], "time": [], "metrics": [],
             "energy_types": [], "normalized_question": question}
    try:
        raw = chat([
            {"role": "system", "content": _ENTITY_SYS},
            {"role": "user", "content": prompt},
        ], temperature=0.0)
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0 or e <= s:
            return empty
        r = json.loads(raw[s:e])
        return {
            "brands": r.get("brands") or [],
            "models": r.get("models") or [],
            "time": r.get("time") or [],
            "metrics": r.get("metrics") or [],
            "energy_types": r.get("energy_types") or [],
            "normalized_question": r.get("normalized_question") or question,
        }
    except Exception:
        return empty


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(
    question: str,
    history: list = None,
    context_block: str = "",
    last_assistant: str = "",
) -> dict:
    """5层 NLU 完整流程。

    Returns dict with keys:
        intent, confidence, entities, is_complete, missing_slots,
        normalized_question, source
    """
    _ = history  # reserved for future multi-turn use; context_block already built by caller

    # Layer 1
    l1 = layer1_prefilter(question, last_assistant)
    if l1 and "intent" in l1:
        return {"intent": l1["intent"], "confidence": l1["confidence"],
                "entities": {}, "is_complete": True, "missing_slots": [],
                "normalized_question": question, "source": l1["source"]}
    force_rag = bool(l1 and l1.get("force_rag"))

    # Layer 2
    l2 = layer2_classify(question, context_block, force_rag=force_rag)

    # Confidence gate → clarify if ambiguous
    if not _confidence_gate(l2):
        return {"intent": "clarify", "confidence": l2["confidence"],
                "entities": {}, "is_complete": False,
                "missing_slots": ["问题意图不明确，请描述得更具体"],
                "normalized_question": question, "source": "confidence_gate"}

    # Layer 3 (only for data-related intents)
    if l2["intent"] in ("sql", "rag", "hybrid"):
        l3 = layer3_extract_entities(question, context_block)
    else:
        l3 = {"brands": [], "models": [], "time": [], "metrics": [],
              "energy_types": [], "normalized_question": question}

    # Layer 5 (before completeness check so forced rag skips slot check)
    final_intent = layer5_business_rules(l2["intent"], question)

    # Layer 4
    is_complete, missing = layer4_check_completeness(final_intent, l3, question)
    if not is_complete and final_intent in ("sql", "hybrid"):
        final_intent = "clarify"

    return {
        "intent": final_intent,
        "confidence": l2["confidence"],
        "entities": {k: l3[k] for k in ("brands", "models", "time", "metrics", "energy_types")},
        "is_complete": is_complete or final_intent not in ("sql", "hybrid"),
        "missing_slots": missing,
        "normalized_question": l3.get("normalized_question", question),
        "source": l2["source"],
    }
```

- [ ] **Step 4: Run all NLU tests**

```bash
.venv/Scripts/pytest tests/test_nlu.py -v
```

Expected: all tests PASS (目标 ~22 tests)

- [ ] **Step 5: Commit**

```bash
git add app/nlu.py tests/test_nlu.py
git commit -m "feat(nlu): layer 3 entity extraction + classify() public API"
```

---

## Task 5: Wire `app/graph.py` to use `nlu.classify()`

**Files:**
- Modify: `app/graph.py`
  - Replace `intent_router` body
  - Add backwards-compat shims for `test_graph.py`
  - Remove `_INTENT_SYSTEM` (no longer needed)

**Interfaces:**
- Consumes: `nlu.classify(question, history, context_block, last_assistant) -> dict`
- The existing `AgentState`, `route_intent`, `route_exec`, `route_verify` are **unchanged**

- [ ] **Step 1: Run existing graph tests to confirm they pass before changes**

```bash
.venv/Scripts/pytest tests/test_graph.py -v 2>&1 | tail -20
```

Expected: all tests PASS (this is the regression baseline)

- [ ] **Step 2: Modify `app/graph.py`**

Replace the `_INTENT_SYSTEM`, `_NO_DATA_SIGNALS`, `_GREETING_PREFIXES`, and `_classify_intent` block (lines ~165-228 in graph.py) with backwards-compat shims:

```python
# ── Backwards-compat shims (tests import these) ───────────────────────────────
# Logic moved to app/nlu.py; these re-export for test_graph.py compatibility.
from .nlu import get_cfg as _get_nlu_cfg

def _get_config_list(key):
    return _get_nlu_cfg().get(key, [])

_GREETING_PREFIXES = tuple(_get_config_list("greeting_prefixes")) or (
    "你好", "您好", "早安", "晚安", "早上好", "再见", "拜拜")
_NO_DATA_SIGNALS = frozenset(_get_config_list("no_data_signals")) or frozenset([
    "未查询到", "没有找到", "未检索到"])

def _classify_intent(question: str, context_block: str) -> dict:
    """Shim: delegates to nlu.classify(). Kept for test_graph.py imports."""
    from .nlu import classify as _nlu_classify
    return _nlu_classify(question=question, context_block=context_block)
```

Replace the `intent_router` function body with:

```python
def intent_router(state: AgentState):
    """意图路由：委托给 nlu.classify()，保留跨轮实体记忆更新。"""
    from .nlu import classify as nlu_classify

    q = state["question"]
    history = state.get("history") or []

    active_ents = state.get("active_entities") or _build_active_entities_from_history(history)
    ctx = _history_block({**state, "active_entities": active_ents})

    last_assistant = ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            last_assistant = (m.get("content") or "")[:300]
            break

    result = nlu_classify(
        question=q,
        history=history,
        context_block=ctx,
        last_assistant=last_assistant,
    )

    intent = result["intent"]
    new_active = _merge_entities(active_ents, result["entities"])

    upd: dict = {
        "intent": intent,
        "confidence": result["confidence"],
        "entities": result["entities"],
        "active_entities": new_active,
        "normalized_question": result["normalized_question"],
        "retry_count": 0,
        "trace": [_t("intent_router",
                     intent=intent,
                     confidence=round(result["confidence"], 2),
                     entities=result["entities"],
                     source=result.get("source", ""))],
    }
    if intent == "clarify":
        upd["clarify_question"] = _build_clarify_question_from_slots(result.get("missing_slots") or [])
    return upd
```

Also remove the old `_build_clarify_question` function and add this replacement:

```python
def _build_clarify_question_from_slots(slots: list) -> str:
    base = "您的问题信息不够完整，需要补充以下内容才能给出准确答案：\n\n"
    if slots:
        base += "\n".join(f"· {s}" for s in slots) + "\n\n"
    base += (
        "示例完整问法：\n"
        "✓「2025年纯电销量Top10」\n"
        "✓「比亚迪各车系今年销量对比」\n"
        "✓「理想L6和小米SU7谁卖得多」"
    )
    return base
```

- [ ] **Step 3: Run graph tests to verify no regression**

```bash
.venv/Scripts/pytest tests/test_graph.py -v
```

Expected: all tests still PASS

- [ ] **Step 4: Run full test suite**

```bash
.venv/Scripts/pytest tests/ -v --ignore=tests/test_rag.py --ignore=tests/test_api.py -x 2>&1 | tail -30
```

Expected: no new failures (test_rag.py and test_api.py need running services, skip for now)

- [ ] **Step 5: Commit**

```bash
git add app/graph.py
git commit -m "feat(nlu): wire intent_router to nlu.classify(), add backwards-compat shims"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|-------------|------|
| Layer 1: rule pre-filter (greeting bypass) | Task 2 |
| Layer 2: LLM classify single-task | Task 3 |
| Layer 2: few-shot vector retrieval | Task 3 |
| Layer 2: confidence threshold gate | Task 2 |
| Layer 2: ambiguity gap gate | Task 2 |
| Layer 3: entity extraction separate call | Task 4 |
| Layer 4: slot completeness rules | Task 2 |
| Layer 5: business rules from YAML | Task 2 |
| Config-driven (YAML) | Task 1 |
| graph.py intent_router wired | Task 5 |
| Backwards compat for test_graph.py | Task 5 |
| Tests for all layers | Tasks 2-5 |

**Placeholder scan:** None found — all steps contain concrete code.

**Type consistency:** `classify()` returns dict with consistent keys used in Task 5's `intent_router`.

---

Plan complete and saved to `docs/superpowers/plans/2026-07-06-enterprise-nlu.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks

**2. Inline Execution** — execute tasks in this session using executing-plans

Which approach?
