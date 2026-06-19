"""LangGraph 双脑编排（PRD-2 §7）：用状态图把 数据脑(Text2SQL) 与 知识脑(RAG) 真正编排起来。

为什么用状态图而非 if-else（§7.1）：
- 显式 State + 节点 + 条件边，复杂多步流程可控可回溯；
- 天然表达「环」——Text2SQL 自校验重试是带环流程（exec_sql ⇄ fix_sql ⇄ verify_sql）；
- 支持「澄清-续问」——clarify 节点反问后结束本轮（→END）；用户的补充作为带 history 的**新一轮请求**
  进来，intent_router 借 history 理解指代后重新路由（**对话级澄清**，状态持久化在 message 表/前端会话）；
- 支持「并行 join」——hybrid 并行跑两个脑链，compose 合并；
- 可观测——每个节点把决策/SQL/检索/重试写入 State.trace。

> 关于「图级挂起/恢复」（面试高频追问）：LangGraph 提供 checkpointer（MemorySaver/PostgresSaver）+ interrupt()
> 做真正的图内断点恢复。本项目**刻意没用**——澄清是对话级的（状态已落 message 表 + 前端会话），再叠一层
> LangGraph 检查点是重复持久化，且 SSE 流式下管理恢复点复杂度高、收益低。**何时该上**：若要做图内多轮
> 挂起（如分步收集多个查询槽位、Human-in-the-loop 审批），再引入 checkpointer + thread_id=conversation_id。

状态流转（§7.5）：
  intent_router ─┬─ sql    → schema_link → gen_sql → exec_sql ─┬─成功→ chart → insight ┐
                 │                                   ▲          ├─失败可重试→ fix_sql ─┘(回 exec_sql)
                 │                                   └──────────┴─失败耗尽→ insight(降级)
                 ├─ rag    → rag_retrieve → rag_answer ───────────────────────────────┐
                 ├─ hybrid → [schema_link 链 ∥ rag_retrieve 链]（并行）                │
                 └─ clarify→ clarify → END                                            ▼
                                              insight / rag_answer ───────────────→ compose → END
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from .llm import chat
from .config import MAX_SQL_RETRY
from .schema_linking import link_schema
from .sql_guard import ensure_safe, with_limit, UnsafeSQLError
from .db import run_query
from .text2sql import DOMAIN, FEWSHOT, SYS as SQL_SYS, _extract_sql
from .charts import recommend_chart


# ============================================================ State（§7.2）
class AgentState(TypedDict, total=False):
    question: str                       # 当前问题
    history: list                       # 多轮上下文
    user_id: int                        # RAG 多租户隔离用
    intent: str                         # sql / rag / hybrid / clarify
    clarify_question: str               # 需澄清时的反问
    linked_schema: str                  # Schema Linking 结果
    sql: str                            # 生成的 SQL
    cols: list                          # 结果列
    rows: list                          # 结果行
    sql_error: str                      # 执行/校验错误
    retry_count: int                    # 已重试次数
    sql_verified: bool                  # 语义自校验结果（结果是否真的回答了问题）
    chunks: list                        # RAG 归并后的父块上下文
    citations: list                     # 引用
    has_answer: bool                    # RAG 是否有依据
    chart: dict                         # 图表描述符
    insight: str                        # 洞察归因
    rag_answer: str                     # 知识脑答案
    final_answer: str                   # 汇总答案
    degraded: bool                      # SQL 重试耗尽降级标记
    trace: Annotated[list, operator.add]  # 每步留痕（并行分支用 add 合并）


def _t(node, **kw):
    """生成一条 trace。"""
    return {"node": node, **kw}


def _history_block(state) -> str:
    """把近期对话压成一小段上下文，让 LLM 能理解指代（如『那丰田呢』）。只取最近几轮、各截断。"""
    h = state.get("history") or []
    if not h:
        return ""
    lines = []
    for m in h[-4:]:
        role = "用户" if m.get("role") == "user" else "助手"
        c = (m.get("content") or "").strip().replace("\n", " ")[:120]
        if c:
            lines.append(f"{role}：{c}")
    if not lines:
        return ""
    return "近期对话（仅用于理解本轮问题的指代/省略，不要直接回答历史问题）：\n" + "\n".join(lines) + "\n\n"


# ============================================================ 节点（§7.3）
# 只把「几乎不会在闲聊里出现」的强领域词放进快速通道；像「多少/对比/怎么样/为什么/原因」太泛，
# 会把『你多少岁』『今天天气怎么样』误判成 sql/rag，故移出规则、交给 LLM 兜底分类（更准，代价仅一次调用）。
_SQL_KW = ("销量", "排名", "卖", "trend", "趋势", "环比", "同比", "top", "前十", "前5", "几辆", "占比")
_RAG_KW = ("报告", "政策", "怎么看", "解读", "口碑", "观点", "分析师", "续航怎么", "评测")
# 明显的问候/闲聊/超纲词：命中且无领域信号 → 直接友好兜底，连 LLM 都不调（省成本）
_CHAT_KW = ("你好", "您好", "在吗", "吃饭", "谢谢", "多谢", "再见", "拜拜", "晚安", "早安",
            "笑话", "你是谁", "你叫", "无聊", "天气", "几点", "哈哈", "傻", "爱你")


def intent_router(state: AgentState):
    """意图识别：规则前置（命中即跳过 LLM 省成本）→ 否则 LLM few-shot 分类（§7.4）。
    新增 chat 意图：问候/闲聊/与车市无关的问题（如『你吃饭了吗』）友好兜底，
    不再因「LLM 兜底默认 sql」而被拖去硬生成 SQL、画出莫名其妙的图。"""
    q = state["question"]
    has_sql = any(k in q for k in _SQL_KW)
    has_rag = any(k in q for k in _RAG_KW)
    if has_sql and has_rag:
        intent = "hybrid"
    elif has_sql:
        intent = "sql"
    elif has_rag:
        intent = "rag"
    elif any(k in q for k in _CHAT_KW):
        intent = "chat"                              # 明显闲聊/超纲且无领域信号 → 友好兜底（省一次 LLM）
    else:
        # LLM few-shot 兜底分类（含 chat / clarify；默认 clarify 而非瞎猜 sql，§7.4）
        out = chat([
            {"role": "system", "content":
             "判断问题意图，只回一个词：\n"
             "sql = 查数据/统计/排名/趋势（如『Top10 销量』『比亚迪卖了多少』）\n"
             "rag = 问文档/政策/口碑/为什么（如『报告对渗透率怎么预测』『这车口碑如何』）\n"
             "hybrid = 既要数据又要解读（如『比亚迪销量趋势如何，行业怎么看』）\n"
             "chat = 问候/闲聊/与新能源汽车市场无关（如『你好』『你吃饭了吗』『讲个笑话』『你是谁』）\n"
             "clarify = 与车市相关但信息不足/口径不清（如『哪个车好』——好指销量？口碑？价格？不清楚）\n"
             "结合近期对话理解指代（如上轮问销量、本轮『那丰田呢』应判 sql）。\n"
             "只回 sql / rag / hybrid / chat / clarify 之一。"},
            {"role": "user", "content": _history_block(state) + q},
        ], temperature=0.0).strip().lower()
        intent = next((k for k in ("hybrid", "clarify", "chat", "sql", "rag") if k in out), "clarify")
    upd = {"intent": intent, "retry_count": 0,
           "trace": [_t("intent_router", intent=intent, rule_hit=has_sql or has_rag)]}
    if intent == "clarify":
        upd["clarify_question"] = "你的问题信息不足，请补充：想了解销量数据，还是文档/政策解读？具体哪个车系或品牌？"
    return upd


def chitchat(state: AgentState):
    """问候/闲聊/超纲 → 友好说明能力边界、引导回车市话题（→ END）。不查数据、不出图。"""
    return {"final_answer":
            "我是「车市镜」——专注新能源汽车销量数据分析与行业知识问答的助手，暂时只聊车市相关的话题～\n"
            "你可以这样问我：\n"
            "· 数据：『2025年纯电销量 Top10』『比亚迪各车系今年卖了多少』\n"
            "· 解读：『小米SU7 口碑怎么样』『最近的购车补贴政策怎么说』",
            "trace": [_t("chitchat")]}


def clarify(state: AgentState):
    """信息不足反问，挂起等用户（→ END）。"""
    return {"final_answer": state.get("clarify_question", "请补充更多信息。"),
            "trace": [_t("clarify")]}


def schema_link(state: AgentState):
    """筛相关表 DDL（§4.1）。"""
    schema = link_schema(state["question"])
    return {"linked_schema": schema, "trace": [_t("schema_link", tables_chars=len(schema))]}


def _gen_sql_messages(state):
    return [
        {"role": "system", "content": SQL_SYS},
        {"role": "user", "content": f"{DOMAIN}\n\n{FEWSHOT}\n可用表结构:\n{state['linked_schema']}\n\n"
                                    f"{_history_block(state)}Q: {state['question']}\nSQL:"},
    ]


def gen_sql(state: AgentState):
    """生成 SQL（§4.2/4.3）。"""
    sql = _extract_sql(chat(_gen_sql_messages(state), temperature=0.0))
    return {"sql": sql, "trace": [_t("gen_sql", sql=sql)]}


def exec_sql(state: AgentState):
    """护栏校验 + 只读执行（§4.4）。成功写 cols/rows，失败写 sql_error。"""
    try:
        safe = with_limit(ensure_safe(state["sql"]))
        cols, rows = run_query(safe)
        return {"sql": safe, "cols": cols, "rows": rows, "sql_error": None,
                "trace": [_t("exec_sql", ok=True, rows=len(rows))]}
    except (UnsafeSQLError, Exception) as e:
        return {"sql_error": str(e), "trace": [_t("exec_sql", ok=False, error=str(e)[:120])]}


def fix_sql(state: AgentState):
    """把错误回喂模型修正 SQL（自校验重试环，§4.5）。"""
    msgs = _gen_sql_messages(state) + [
        {"role": "assistant", "content": state.get("sql", "")},
        {"role": "user", "content": f"上面的 SQL 执行报错：{state['sql_error']}\n请修正后只输出一条 SELECT。"},
    ]
    sql = _extract_sql(chat(msgs, temperature=0.0))
    n = state.get("retry_count", 0) + 1
    return {"sql": sql, "retry_count": n, "trace": [_t("fix_sql", attempt=n, sql=sql)]}


_VERIFY_SYS = (
    "你是 SQL 审核员。给你一个『自然语言问题』和这条 SQL『实际查询结果(列+前几行)』，"
    "判断结果是否真的回答了问题——重点看：过滤口径(品牌/车系/时间/能源类型)有没有错、"
    "聚合维度对不对、是不是答非所问。结果为空但问题本身合理(可能就是没数据)也算 ok=true。"
    "只输出 JSON：{\"ok\": true/false, \"reason\": \"简短中文原因\"}，不要解释、不要代码块。"
)


def verify_sql(state: AgentState):
    """Text2SQL 语义自校验（§4.6）：SQL 能跑通≠语义对。把『能跑但答非所问』纳入闭环。
    FAIL-OPEN：开关关闭 / 无数据 / 已无重试预算 / 校验自身异常 —— 一律放行，绝不比不校验更差。"""
    from .config import SEMANTIC_CHECK
    rows = state.get("rows") or []
    # 不校验的情形：开关关 / 空结果(无可核对) / 重试预算已耗尽(再判也回不了 fix_sql)
    if not SEMANTIC_CHECK or not rows or state.get("retry_count", 0) >= MAX_SQL_RETRY:
        return {"sql_verified": True, "trace": [_t("verify_sql", checked=False)]}
    try:
        out = chat([
            {"role": "system", "content": _VERIFY_SYS},
            {"role": "user", "content": f"问题：{state['question']}\nSQL：{state.get('sql', '')}\n"
                                        f"列：{state.get('cols')}\n结果(前5行)：{str(rows[:5])}"}],
            temperature=0.0)
        import json as _json
        m = _json.loads(out[out.find("{"):out.rfind("}") + 1])
        if bool(m.get("ok", True)):
            return {"sql_verified": True, "trace": [_t("verify_sql", ok=True)]}
        # 语义不匹配 → 把「为什么不匹配」当错误回喂 fix_sql 重生成（复用现成重试环）
        return {"sql_verified": False,
                "sql_error": f"结果未正确回答问题：{m.get('reason', '口径/过滤/聚合可能有误')}。请修正 SQL。",
                "trace": [_t("verify_sql", ok=False, reason=str(m.get("reason"))[:80])]}
    except Exception as e:  # FAIL-OPEN：校验出任何问题都放行正常结果
        return {"sql_verified": True, "trace": [_t("verify_sql", error=str(e)[:80])]}


def chart(state: AgentState):
    """规则引擎产出图表描述符（§8）。无数据则跳过。"""
    rows = state.get("rows", [])
    if not rows:
        return {"chart": None, "trace": [_t("chart", skipped="no_rows")]}
    spec = recommend_chart(state.get("cols", []), rows, state["question"])
    return {"chart": spec, "trace": [_t("chart", default_type=(spec or {}).get("default_type"))]}


_INSIGHT_SYS = ("你是商业分析顾问。根据问题与查询结果给出：1)一句话结论 2)简要归因 3)1-2 条建议。"
                "只依据给定数据，不编造数字。中文简洁。")


def insight(state: AgentState):
    """洞察归因；无数据/重试耗尽/LLM 降级三条路径。"""
    rows = state.get("rows", []) or []
    cols = state.get("cols", []) or []

    # 路径A：SQL 重试耗尽
    if state.get("sql_error") and not rows:
        return {"degraded": True,
                "insight": "抱歉，这个问题我多次尝试都没能生成可用的查询，可能是口径不清或超出当前数据范围。"
                           "可换个问法或缩小范围（如指定车系/月份）。",
                "trace": [_t("insight", degraded=True)]}

    # 路径B：SQL 执行成功但结果为空 —— 不调 LLM，直接返回明确消息
    if not rows:
        q = state.get("question", "")
        brand_hint = ""
        for word in ["奔驰", "宝马", "奥迪", "丰田", "本田", "大众", "特斯拉", "蔚来", "理想", "小鹏", "比亚迪", "吉利", "长安", "奇瑞", "长城"]:
            if word in q:
                brand_hint = f"「{word}」可能不在当前数据库覆盖范围内（目前覆盖 101 个品牌，以国产新能源为主）。"
                break
        return {"insight": f"未查询到相关数据。{brand_hint}请尝试：\n"
                           f"1. 换一个品牌或车系名称（如 '比亚迪'、'小米SU7'）\n"
                           f"2. 问更宽泛的问题（如 '2025年纯电销量Top10'）",
                "trace": [_t("insight", empty_result=True)]}

    # 路径C：正常结果 → LLM 生成洞察
    txt = chat([{"role": "system", "content": _INSIGHT_SYS},
                {"role": "user", "content": f"问题：{state['question']}\n列：{cols}\n"
                                            f"数据(前20行)：{str(rows[:20])}"}],
               temperature=0.3)
    return {"insight": txt, "trace": [_t("insight", chars=len(txt))]}


def rag_retrieve(state: AgentState):
    """知识脑：混合召回 + 重排 + 父块归并（§5.4）。"""
    from .rag import retrieve as R
    children = R.hybrid_recall(state["user_id"], state["question"])
    if not children:
        return {"chunks": [], "trace": [_t("rag_retrieve", recall=0)]}
    top, used_rr = R.rerank(state["question"], children)
    top_score = top[0]["score_final"] if top else 0.0
    from .config import RERANK_SCORE_MIN
    if used_rr and top_score < RERANK_SCORE_MIN:
        return {"chunks": [], "trace": [_t("rag_retrieve", recall=len(children), low_score=round(top_score, 3))]}
    blocks = R.merge_parents(top)
    return {"chunks": blocks,
            "trace": [_t("rag_retrieve", recall=len(children), parents=len(blocks),
                         reranker=used_rr, top_score=round(top_score, 3))]}


def rag_answer(state: AgentState):
    """知识脑：带引用生成 + 防幻觉（§5.5）。"""
    from .rag import retrieve as R
    blocks = state.get("chunks") or []
    if not blocks:
        return {"rag_answer": R.NO_ANSWER, "citations": [], "has_answer": False,
                "trace": [_t("rag_answer", has_answer=False)]}
    res = R.generate(state["question"], blocks)
    return {"rag_answer": res["answer"], "citations": res["citations"],
            "has_answer": res["has_answer"], "trace": [_t("rag_answer", has_answer=res["has_answer"],
                                                          citations=len(res["citations"]))]}


def compose(state: AgentState):
    """汇总：把 数据结论(图表+洞察) 与 文档佐证(带引用答案) 拼成统一回答（§7.6）。任一脑缺失则优雅降级。"""
    parts = []
    # 数据脑：只有有真实数据时才标注【数据分析】
    if state.get("insight"):
        has_data = bool(state.get("rows"))
        prefix = "【数据分析】" if (has_data and state.get("rag_answer")) else ""
        parts.append(prefix + state["insight"])
    if state.get("rag_answer"):
        parts.append(("【文档佐证】" if parts else "") + state["rag_answer"])
    final = "\n\n".join(p for p in parts if p) or "未能生成回答。"
    return {"final_answer": final, "trace": [_t("compose",
            has_chart=bool(state.get("chart")), has_citations=bool(state.get("citations")))]}


# ============================================================ 条件路由（§7.5）
def route_intent(state: AgentState):
    intent = state["intent"]
    if intent == "sql":
        return "schema_link"
    if intent == "rag":
        return "rag_retrieve"
    if intent == "hybrid":
        return ["schema_link", "rag_retrieve"]    # 并行 fan-out（两个脑链同时跑）
    if intent == "chat":
        return "chitchat"
    return "clarify"


def route_exec(state: AgentState):
    if not state.get("sql_error"):
        return "verify_sql"                         # 成功 → 先语义自校验（再出图）
    if state.get("retry_count", 0) < MAX_SQL_RETRY:
        return "fix_sql"                            # 失败可重试 → 修正环
    return "insight"                                # 重试耗尽 → 降级（insight 节点出降级话术）


def route_verify(state: AgentState):
    if state.get("sql_verified", True):
        return "chart"                              # 语义 OK → 出图
    return "fix_sql"                                # 语义不匹配 → 回 fix_sql 重生成（verify 已确认仍有预算）


# ============================================================ 建图
def build_graph():
    g = StateGraph(AgentState)
    for fn in (intent_router, clarify, chitchat, schema_link, gen_sql, exec_sql, fix_sql, verify_sql,
               chart, insight, rag_retrieve, rag_answer):
        g.add_node(fn.__name__, fn)
    # compose 是「并行 join」节点：defer=True 让它延到所有在途分支(数据脑链+知识脑链)都完成才跑一次，
    # 避免 hybrid 下两条链长度不一导致 compose 被触发两次（§7.6 合并语义）。
    g.add_node("compose", compose, defer=True)

    g.add_edge(START, "intent_router")
    g.add_conditional_edges("intent_router", route_intent,
                            ["schema_link", "rag_retrieve", "clarify", "chitchat"])
    # 数据脑链
    g.add_edge("schema_link", "gen_sql")
    g.add_edge("gen_sql", "exec_sql")
    g.add_conditional_edges("exec_sql", route_exec, ["verify_sql", "fix_sql", "insight"])
    g.add_conditional_edges("verify_sql", route_verify, ["chart", "fix_sql"])  # 语义校验 → 出图 / 回修
    g.add_edge("fix_sql", "exec_sql")               # 重试环（执行报错 + 语义不匹配 共用）
    g.add_edge("chart", "insight")
    g.add_edge("insight", "compose")
    # 知识脑链
    g.add_edge("rag_retrieve", "rag_answer")
    g.add_edge("rag_answer", "compose")
    # 收口
    g.add_edge("compose", END)
    g.add_edge("clarify", END)
    g.add_edge("chitchat", END)
    return g.compile()


_GRAPH = None


def run_agent(question: str, user_id: int, history: list = None) -> dict:
    """对外入口：跑图，返回最终 State（含 final_answer / chart / citations / trace）。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    init = {"question": question, "user_id": user_id, "history": history or [], "trace": []}
    return _GRAPH.invoke(init)


def stream_agent(question: str, user_id: int, history: list = None):
    """流式跑图：用 LangGraph `.stream(values)` 逐超步产出**累积 State 快照**。
    供 SSE 层在某节点完成、对应字段一出现就立刻推事件（渐进反馈，告别『等 5-10s 一次性全出』）。
    最后一个 yield 的快照即最终 State。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    init = {"question": question, "user_id": user_id, "history": history or [], "trace": []}
    for snapshot in _GRAPH.stream(init, stream_mode="values"):
        yield snapshot
