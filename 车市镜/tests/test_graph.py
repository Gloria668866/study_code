"""LangGraph 编排关键路径单测（Agent）：意图规则前置 + 条件路由（均不调 LLM，确定性）。"""
from app.graph import (route_intent, route_exec, route_verify, verify_sql,
                       intent_router, chitchat, _SQL_KW, _RAG_KW)
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


def test_route_intent_chat():
    assert route_intent({"intent": "chat"}) == "chitchat"


# —— 闲聊/超纲意图（修复『你吃饭了吗→出图』）：命中闲聊词、无领域信号 → chat，不调 LLM —— #
def test_intent_chitchat_eat():
    # 用户实测的 bug：『你吃饭了吗』曾被默认当 sql 去出图
    assert intent_router({"question": "你吃饭了吗"})["intent"] == "chat"


def test_intent_chitchat_greeting():
    assert intent_router({"question": "你好"})["intent"] == "chat"


def test_intent_chitchat_who():
    assert intent_router({"question": "你是谁"})["intent"] == "chat"


def test_chitchat_node_no_chart():
    # 闲聊兜底只给文案、绝不带 chart/sql/rows
    out = chitchat({"question": "你吃饭了吗"})
    assert out.get("final_answer") and "chart" not in out and "sql" not in out


# —— 关键词瘦身：泛词不再误命中规则（这些应落到 LLM 兜底，而非被规则直接判错） —— #
def test_no_false_positive_age():
    assert not any(k in "你多少岁" for k in _SQL_KW)          # 移除『多少』后不再误判 sql


def test_no_false_positive_mood():
    assert not any(k in "帮我对比一下心情" for k in _SQL_KW)    # 移除『对比』后不再误判 sql


def test_no_false_positive_weather():
    assert not any(k in "今天天气怎么样" for k in _RAG_KW)      # 移除『怎么样』后不再误判 rag


def test_route_exec_success_to_verify():
    assert route_exec({"retry_count": 0}) == "verify_sql"      # 无 sql_error → 先语义自校验


def test_route_exec_retry():
    assert route_exec({"sql_error": "boom", "retry_count": 0}) == "fix_sql"


def test_route_exec_giveup_to_insight():
    assert route_exec({"sql_error": "boom", "retry_count": MAX_SQL_RETRY}) == "insight"


# —— 语义自校验路由 + fail-open（不调 LLM 的分支都得确定性放行） —— #
def test_route_verify_ok_to_chart():
    assert route_verify({"sql_verified": True}) == "chart"


def test_route_verify_mismatch_to_fix():
    assert route_verify({"sql_verified": False}) == "fix_sql"


def test_route_verify_default_failopen():
    assert route_verify({}) == "chart"                        # 缺字段默认放行（不拦正常结果）


def test_verify_sql_skips_when_no_rows():
    # 空结果无可核对 → 不调 LLM、直接放行
    out = verify_sql({"question": "x", "rows": []})
    assert out["sql_verified"] is True


def test_verify_sql_skips_when_budget_exhausted():
    # 重试预算耗尽 → 再判也回不了 fix_sql，直接放行（不调 LLM）
    out = verify_sql({"question": "x", "rows": [(1,)], "retry_count": MAX_SQL_RETRY})
    assert out["sql_verified"] is True


# —— 意图规则前置（命中关键词即不调 LLM） —— #
def test_intent_rule_sql():
    assert intent_router({"question": "2025年纯电销量排名前十"})["intent"] == "sql"


def test_intent_rule_rag():
    assert intent_router({"question": "这份报告怎么看渗透率"})["intent"] == "rag"


def test_intent_rule_hybrid():
    assert intent_router({"question": "比亚迪销量趋势如何，行业怎么看前景"})["intent"] == "hybrid"
