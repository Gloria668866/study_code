"""LangGraph 编排关键路径单测（Agent）：意图规则前置 + 条件路由（均不调 LLM，确定性）。"""
from app.graph import route_intent, route_exec, intent_router
from app.config import MAX_SQL_RETRY


# —— 条件路由（纯函数） —— #
def test_route_intent_sql():
    assert route_intent({"intent": "sql"}) == "schema_link"


def test_route_intent_rag():
    assert route_intent({"intent": "rag"}) == "rag_retrieve"


def test_route_intent_hybrid_fanout():
    assert route_intent({"intent": "hybrid"}) == ["schema_link", "rag_retrieve"]


def test_route_intent_clarify():
    assert route_intent({"intent": "clarify"}) == "clarify"


def test_route_exec_success_to_chart():
    assert route_exec({"retry_count": 0}) == "chart"          # 无 sql_error → 出图


def test_route_exec_retry():
    assert route_exec({"sql_error": "boom", "retry_count": 0}) == "fix_sql"


def test_route_exec_giveup_to_insight():
    assert route_exec({"sql_error": "boom", "retry_count": MAX_SQL_RETRY}) == "insight"


# —— 意图规则前置（命中关键词即不调 LLM） —— #
def test_intent_rule_sql():
    assert intent_router({"question": "2025年纯电销量排名前十"})["intent"] == "sql"


def test_intent_rule_rag():
    assert intent_router({"question": "这份报告怎么看渗透率"})["intent"] == "rag"


def test_intent_rule_hybrid():
    assert intent_router({"question": "比亚迪销量趋势如何，行业怎么看前景"})["intent"] == "hybrid"
