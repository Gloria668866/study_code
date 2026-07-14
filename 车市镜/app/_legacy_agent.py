"""Agent 编排：意图识别 → 路由 → (查数据 / 查文档) → 图表 → 洞察。

本骨架先用「线性编排」跑通（零额外依赖）。M2 升级为 LangGraph StateGraph：
每个函数即一个节点，路由用条件边，状态在节点间传递、可观测。
节点映射见 PRD 图 2.3。
"""
from .llm import chat
from .text2sql import nl_to_sql_and_run
from .charts import recommend_chart


def classify_intent(question: str) -> str:
    """返回 'sql' | 'doc' | 'chat'。doc 路径（报告 RAG）在 M2 接入。"""
    out = chat([
        {"role": "system", "content": "判断问题类型，只回一个词：sql(需查数据库) / doc(问口径或报告) / chat(闲聊)。"},
        {"role": "user", "content": question},
    ], temperature=0.0).strip().lower()
    for k in ("sql", "doc", "chat"):
        if k in out:
            return k
    return "sql"


def analyze(question: str) -> dict:
    """非流式：返回意图、SQL、数据、图表规格。洞察走 main 的流式接口。"""
    intent = classify_intent(question)
    if intent != "sql":
        return {"intent": intent, "sql": None, "cols": [], "rows": [],
                "chart": None, "note": "doc/chat 路径将在 M2 接入报告 RAG"}
    res = nl_to_sql_and_run(question)
    # 图表用规则引擎产出「描述符」（不再调 LLM），question 作标题；前端配合 rows 自由拼图
    chart = recommend_chart(res["cols"], res["rows"], question) if res["rows"] else None
    return {"intent": "sql", **res, "chart": chart}
