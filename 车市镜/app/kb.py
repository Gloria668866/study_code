"""知识库 API（RAG，PRD-2 §10）：上传 → 异步解析入库 → 列表/状态 → 软删除 → 问答。
【全部需登录，按 user_id 隔离】

与登录模块旧 /api/kb/list（读 SQLite app.db 占位）不同，这里的 kb_document/kb_chunk 落 PostgreSQL
（pgvector，config.RAG_DATABASE_URL）。生产把 APP_DATABASE_URL 也指向同一 PG 即统一两边。

上传是「轻返回 + 重后台」：接口只 存MinIO + 建档(parsing) + 投 Celery，立即返回 doc_id；
worker 异步解析/切块/向量化，前端轮询 GET /api/kb/{id} 看 status(parsing→ready/failed)。

注意：kb_document.user_id 外键指向 PG users(id)，故 app 需以 APP_DATABASE_URL=PG 运行
（登录用户落在 PG），上传才能通过外键。纯管线验收见 data/rag_ingest_demo.py（不经 HTTP）。
"""
import os

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel

from .auth import get_current_user
from .models import User
from .rag import pg, ingest
from .rag.tasks import ingest_document_task

router = APIRouter(prefix="/api/kb", tags=["kb"])

_EXT2TYPE = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "md", ".txt": "text"}
MAX_UPLOAD_MB = 20                              # 单文件大小上限


class AskIn(BaseModel):
    question: str


@router.post("/upload")
async def upload(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    """上传文档建知识库：校验类型/大小 → 存 MinIO + 建 kb_document(parsing) + 投异步解析任务。"""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _EXT2TYPE:                    # 类型校验（结构化错误）
        raise HTTPException(415, f"不支持的文件类型 {ext or '(无扩展名)'}；仅支持 {'/'.join(sorted(_EXT2TYPE))}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "空文件")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:  # 大小校验
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD_MB}MB 上限")
    doc_id = ingest.stage_for_async(user.id, file.filename, data, _EXT2TYPE[ext], title=file.filename)
    ingest_document_task.delay(doc_id)           # 异步：解析/切块/向量化在 worker 跑
    return {"doc_id": doc_id, "status": "parsing", "file_type": _EXT2TYPE[ext]}


@router.get("/list")
def kb_list(user: User = Depends(get_current_user)):
    """当前用户的知识库文档列表（已过滤软删除，含解析状态）。"""
    return {"documents": pg.list_documents(user.id)}


@router.get("/{doc_id}")
def document_status(doc_id: int, user: User = Depends(get_current_user)):
    """单个文档状态（前端轮询 parsing→ready/failed）。只能看自己的。"""
    doc = pg.get_document(doc_id)
    if doc is None or doc["user_id"] != user.id or doc.get("deleted_at"):
        raise HTTPException(404, "文档不存在")
    return {"doc_id": doc["id"], "filename": doc["filename"], "status": doc["status"],
            "chunk_count": doc["chunk_count"], "file_type": doc["file_type"]}


@router.delete("/{doc_id}")
def kb_delete(doc_id: int, user: User = Depends(get_current_user)):
    """软删除文档：之后检索/列表都不再含它（按 user_id 校验归属）。"""
    doc = pg.get_document(doc_id)
    if doc is None or doc["user_id"] != user.id or doc.get("deleted_at"):
        raise HTTPException(404, "文档不存在")
    pg.soft_delete(doc_id)
    return {"doc_id": doc_id, "deleted": True}


@router.post("/ask")
def ask(body: AskIn, user: User = Depends(get_current_user)):
    """RAG 在线问答：检索→归并→带引用生成（§5.4/5.5）。只检索当前用户自己的文档（隔离）。
    返回 {answer, citations:[{doc_id,page_no,chunk_id,...}], has_answer}。"""
    from .rag.retrieve import answer_question
    if not body.question.strip():
        raise HTTPException(400, "问题不能为空")
    return answer_question(user.id, body.question.strip())
