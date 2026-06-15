"""FastAPI 入口（接口清单见 PRD-2 §10；SSE 事件协议见 §9.1）：
- /api/auth/register|login|me            鉴权（app/auth.py）
- POST /api/ask        (SSE)             双脑问答，按 §9.1 推 intent/sql/rows/chart/insight/citation/done/error
- POST /api/ask_sync                     同步返回完整结果（调试）
- GET  /api/history                      当前用户会话列表
- GET  /api/history/{conv_id}            会话消息（还原含图表/引用的历史会话）
- POST /api/kb/upload | GET /api/kb/list | DELETE /api/kb/{doc_id} | POST /api/kb/ask  （app/kb.py）

鉴权与隔离（§17.3）：除 /health 与 /api/auth/* 外，业务接口都 Depends(get_current_user)；
落库/查询一律带当前用户 user_id —— A 看不到 B 的会话与知识库。
编排：/api/ask 走 LangGraph 双脑状态图（app/graph.run_agent，§7），不再用旧的线性 analyze。
"""
import asyncio
import json
import logging
import secrets
import threading

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from .graph import run_agent, stream_agent
from .auth import router as auth_router, get_current_user
from .database import get_db, init_db
from .models import User, Conversation, Message, SavedInsight, SharedInsight

from .config import CORS_ALLOW_ORIGINS, JWT_SECRET, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger("cheshijing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="车市镜 · 新能源车市情报 Agent")
# CORS：dev 默认放行全部；生产用 CORS_ALLOW_ORIGINS 收紧到正式域名（§17/§18）。
# 用 Bearer token 鉴权（非 cookie），故 "*" 时不需 credentials（浏览器禁止 * + credentials 同用）。
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOW_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=(CORS_ALLOW_ORIGINS != ["*"]))
app.include_router(auth_router)
# RAG 知识库路由（上传/列表/删除/问答，PG+pgvector）。延迟导入避免无 RAG 依赖时启动失败。
try:
    from .kb import router as kb_router
    app.include_router(kb_router)
except Exception as _e:  # noqa: BLE001
    print(f"[warn] RAG kb 路由未加载（缺依赖或 PG 未起）：{_e}")


@app.on_event("startup")
def _startup():
    init_db()  # 幂等建应用层表（users/conversation/message/kb_document/saved_insight/shared_insight）
    # 安全自检：JWT 弱密钥在生产环境是致命的（任何人可伪造任意用户 token）。
    if JWT_SECRET == "dev-insecure-change-me":
        logger.warning("JWT_SECRET 仍是默认弱密钥！上线/演示前务必在 .env 设置随机强密钥（如 openssl rand -hex 32）")
    # 配置自检：base_url 与 model 跨厂商不一致 = 静默连错 API（O7：qwen 默认值 + deepseek base 的坑）。
    _base, _model = LLM_BASE_URL.lower(), LLM_MODEL.lower()
    _provider = next((p for p in ("deepseek", "dashscope", "moonshot", "openai", "siliconflow") if p in _base), None)
    if _provider == "deepseek" and "deepseek" not in _model:
        logger.warning(f"LLM 配置可能不匹配：LLM_BASE_URL 指向 DeepSeek，但 LLM_MODEL='{LLM_MODEL}'（非 deepseek-*）。请核对 .env。")
    elif _provider == "dashscope" and not (_model.startswith("qwen") or _model.startswith("qwq")):
        logger.warning(f"LLM 配置可能不匹配：LLM_BASE_URL 指向 DashScope/通义，但 LLM_MODEL='{LLM_MODEL}'。请核对 .env。")


class Ask(BaseModel):
    question: str
    conversation_id: int | None = None  # 可选：续接已有会话；缺省则新建


@app.get("/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------- 落库
def _row_arrays(state) -> list:
    """结果集 dict 行 → 按列序的数组行（§9.1 rows 事件要数组）。"""
    cols = state.get("cols") or []
    return [[r.get(c) for c in cols] for r in (state.get("rows") or [])]


def _persist(db: Session, user: User, question: str, state: dict,
             conversation_id: int | None) -> tuple[int, int]:
    """把一轮问答落库到当前用户名下；assistant 消息存 result_meta（图表/列行/引用），供历史还原。"""
    conv = None
    if conversation_id is not None:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user.id:    # 只能续接自己的会话
            conv = None
    if conv is None:
        conv = Conversation(user_id=user.id, title=question[:40])
        db.add(conv)
        db.flush()
    db.add(Message(conversation_id=conv.id, user_id=user.id, role="user", content=question))
    meta = {
        "columns": state.get("cols") or [],
        "rows": _row_arrays(state),
        "chart": state.get("chart"),
        "citations": state.get("citations") or [],
        "intent": state.get("intent"),
        "trace": [t.get("node") for t in (state.get("trace") or [])],
    }
    assistant = Message(
        conversation_id=conv.id, user_id=user.id, role="assistant",
        content=(state.get("final_answer") or "")[:4000],
        intent=state.get("intent"), sql_text=state.get("sql"),
        result_meta=json.dumps(meta, ensure_ascii=False, default=str),
    )
    db.add(assistant)
    db.commit()
    return conv.id, assistant.id


# ---------------------------------------------------------------- 问答
@app.post("/api/ask_sync")
def ask_sync(body: Ask, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """同步返回完整结果（调试用）。"""
    state = run_agent(body.question, user.id)
    conv_id, msg_id = _persist(db, user, body.question, state, body.conversation_id)
    return {
        "intent": state.get("intent"), "sql": state.get("sql"),
        "columns": state.get("cols") or [], "rows": _row_arrays(state),
        "chart": state.get("chart"), "answer": state.get("final_answer"),
        "citations": state.get("citations") or [], "has_answer": state.get("has_answer", True),
        "conversation_id": conv_id, "msg_id": msg_id,
        "trace": [t.get("node") for t in (state.get("trace") or [])],
    }


def _insight_pieces(text: str, n: int = 24):
    for i in range(0, len(text), n):
        yield text[i:i + n]


def _load_history(db: Session, user: User, conversation_id: int | None, limit: int = 6) -> list:
    """取本会话最近几轮消息作多轮上下文（仅自己的会话）。供 Agent 理解『那丰田呢』这类指代。"""
    if conversation_id is None:
        return []
    conv = db.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        return []
    msgs = db.scalars(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc()).limit(limit)
    ).all()
    return [{"role": m.role, "content": m.content or ""} for m in reversed(msgs)]


@app.post("/api/ask")
async def ask(body: Ask, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """SSE（§9.1）：stage(accepted) → intent → [sql] → [rows] → [chart] → insight(逐段 delta) → [citation...] → done。
    **渐进推送**：用 LangGraph 流式跑图，某节点一完成、对应字段一出现就立刻推事件（不再等整图跑完一次性全出）。
    纯 RAG 无 sql/rows/chart，只有 insight(答案)+citation；出错推 error。"""
    history = _load_history(db, user, body.conversation_id)  # O3 多轮上下文

    async def gen():
        yield {"event": "stage", "data": json.dumps(
            {"stage": "accepted", "message": "已接收，正在分析问题…"}, ensure_ascii=False)}

        # 同步流式图丢到工作线程跑，用线程安全队列把每个「累积 State 快照」桥接回事件循环。
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker():
            try:
                for snap in stream_agent(body.question, user.id, history):
                    loop.call_soon_threadsafe(q.put_nowait, ("snap", snap))
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, ("err", e))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("end", None))

        threading.Thread(target=worker, daemon=True).start()

        emitted: set = set()
        final: dict = {}
        err = None
        while True:
            kind, payload = await q.get()
            if kind == "err":
                err = payload
                break
            if kind == "end":
                break
            snap = payload
            final = snap
            # 字段一出现就推，每种事件只推一次（渐进反馈的关键）
            if snap.get("intent") and "intent" not in emitted:
                emitted.add("intent")
                yield {"event": "intent", "data": json.dumps(
                    {"intent": snap.get("intent"), "confidence": None}, ensure_ascii=False)}
            if snap.get("sql") and "sql" not in emitted:
                emitted.add("sql")
                yield {"event": "sql", "data": json.dumps({"sql_text": snap["sql"]}, ensure_ascii=False)}
            if snap.get("rows") and "rows" not in emitted:
                emitted.add("rows")
                yield {"event": "rows", "data": json.dumps(
                    {"columns": snap.get("cols") or [], "rows": _row_arrays(snap)},
                    ensure_ascii=False, default=str)}
            if snap.get("chart") and "chart" not in emitted:
                emitted.add("chart")
                yield {"event": "chart", "data": json.dumps(snap["chart"], ensure_ascii=False)}
            if snap.get("final_answer") and "insight" not in emitted:
                emitted.add("insight")
                for piece in _insight_pieces(snap.get("final_answer") or "（无内容）"):
                    yield {"event": "insight", "data": json.dumps({"delta": piece}, ensure_ascii=False)}
                    await asyncio.sleep(0)
            if snap.get("citations") and "citations" not in emitted:
                emitted.add("citations")
                for c in snap.get("citations") or []:
                    yield {"event": "citation", "data": json.dumps(c, ensure_ascii=False, default=str)}

        if err is not None:
            yield {"event": "error", "data": json.dumps(
                {"code": "AGENT_ERROR", "message": "处理失败，请换种问法或缩小范围。"}, ensure_ascii=False)}
            logger.exception(f"[ask] stream_agent 失败 user={user.id} q={body.question[:80]!r}: {err}")
            return

        # 落库（拿到完整最终 State 后）+ 收尾
        conv_id, msg_id = _persist(db, user, body.question, final, body.conversation_id)
        if "insight" not in emitted:  # 极端兜底：没有任何 final_answer 也让前端正常结束
            yield {"event": "insight", "data": json.dumps({"delta": "（无内容）"}, ensure_ascii=False)}
        yield {"event": "done", "data": json.dumps(
            {"msg_id": msg_id, "conversation_id": conv_id, "has_answer": final.get("has_answer", True)},
            ensure_ascii=False)}

    return EventSourceResponse(gen(), ping=15)


# ---------------------------------------------------------------- 历史会话
@app.get("/api/history")
def history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """当前用户的会话列表（只返回自己的，按时间倒序）。"""
    convs = db.scalars(
        select(Conversation).where(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
    ).all()
    return {"conversations": [
        {"id": c.id, "title": c.title, "created_at": c.created_at.isoformat() if c.created_at else None}
        for c in convs
    ]}


@app.get("/api/history/{conv_id}")
def history_detail(conv_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """按会话取消息，还原对话（含图表/引用）。只能看自己的（隔离）。"""
    conv = db.get(Conversation, conv_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "会话不存在")
    msgs = db.scalars(
        select(Message).where(Message.conversation_id == conv_id).order_by(Message.created_at)
    ).all()
    out = []
    for m in msgs:
        meta = json.loads(m.result_meta) if m.result_meta else None
        out.append({"role": m.role, "content": m.content, "intent": m.intent,
                    "chart": (meta or {}).get("chart"), "columns": (meta or {}).get("columns"),
                    "rows": (meta or {}).get("rows"), "citations": (meta or {}).get("citations")})
    return {"conversation_id": conv_id, "title": conv.title, "messages": out}


@app.delete("/api/history/{conv_id}")
def history_delete(conv_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """删除会话（连同其消息），按 user_id 校验归属——只能删自己的。"""
    conv = db.get(Conversation, conv_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "会话不存在")
    db.query(Message).filter(Message.conversation_id == conv_id).delete(synchronize_session=False)
    db.delete(conv)
    db.commit()
    return {"conversation_id": conv_id, "deleted": True}


# ---------------------------------------------------------------- 收藏看板
class InsightIn(BaseModel):
    title: str | None = None
    question: str | None = None
    intent: str | None = None
    payload: str | None = None     # 前端传 JSON 字符串（columns/rows/chart/insight/citations/sql）


def _insight_out(it: SavedInsight) -> dict:
    return {"id": it.id, "title": it.title, "question": it.question, "intent": it.intent,
            "payload": it.payload, "created_at": it.created_at.isoformat() if it.created_at else None}


@app.get("/api/insights")
def insights_list(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """当前用户的收藏洞察（按时间倒序）。"""
    rows = db.scalars(
        select(SavedInsight).where(SavedInsight.user_id == user.id).order_by(SavedInsight.created_at.desc())
    ).all()
    return {"insights": [_insight_out(it) for it in rows]}


@app.post("/api/insights")
def insights_create(body: InsightIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """收藏一条洞察快照到当前用户名下。"""
    it = SavedInsight(user_id=user.id, title=(body.title or "")[:255],
                      question=body.question, intent=body.intent, payload=body.payload)
    db.add(it)
    db.commit()
    db.refresh(it)
    return _insight_out(it)


@app.delete("/api/insights/{insight_id}")
def insights_delete(insight_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """移出看板（按 user_id 校验归属，只能删自己的）。"""
    it = db.get(SavedInsight, insight_id)
    if it is None or it.user_id != user.id:
        raise HTTPException(404, "收藏不存在")
    db.delete(it)
    db.commit()
    return {"id": insight_id, "deleted": True}


# ---------------------------------------------------------------- 一键分享
class ShareIn(BaseModel):
    title: str | None = None
    question: str | None = None
    intent: str | None = None
    payload: str | None = None


@app.post("/api/share")
def share_create(body: ShareIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """生成只读公开分享，返回 token（前端拼成 /s/{token}）。"""
    token = secrets.token_urlsafe(9)
    sh = SharedInsight(token=token, user_id=user.id, title=(body.title or "")[:255],
                       question=body.question, intent=body.intent, payload=body.payload)
    db.add(sh)
    db.commit()
    return {"token": token}


@app.get("/api/public/share/{token}")
def share_public(token: str, db: Session = Depends(get_db)):
    """公开读取分享快照——**不需要登录**（这是对外的营销落地数据源）。"""
    sh = db.scalar(select(SharedInsight).where(SharedInsight.token == token))
    if sh is None:
        raise HTTPException(404, "分享不存在或已过期")
    return {"title": sh.title, "question": sh.question, "intent": sh.intent,
            "payload": sh.payload, "created_at": sh.created_at.isoformat() if sh.created_at else None}
