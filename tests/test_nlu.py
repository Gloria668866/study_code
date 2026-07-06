"""Unit tests for app/nlu.py — layers that don't need LLM."""
import pytest


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
