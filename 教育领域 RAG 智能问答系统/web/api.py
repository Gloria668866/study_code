import json
import os
import sys

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from base import Config, logger

conf = Config()

app = FastAPI(title="EduRAG Interview Demo", version="2.0.0", docs_url=None, redoc_url=None)

_qa_system = None


def get_qa_system():
    global _qa_system
    if _qa_system is None:
        from qa_service import IntegratedQASystem
        _qa_system = IntegratedQASystem()
    return _qa_system


def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def load_ragas_scores():
    result_path = os.path.join(
        project_root,
        "rag_qa",
        "rag_assesment",
        "results",
        "ragas_scores_20260505_140306.json",
    )
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


@app.get("/")
async def index():
    html_path = os.path.join(current_dir, "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/health")
async def health():
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}",
            db_name=conf.MILVUS_DATABASE_NAME,
        )
        collection_exists = client.has_collection(conf.MILVUS_COLLECTION_NAME)
        return {
            "status": "ok" if collection_exists else "warning",
            "milvus": "connected",
            "collection": conf.MILVUS_COLLECTION_NAME,
            "collection_exists": collection_exists,
            "database": conf.MILVUS_DATABASE_NAME,
            "llm_model": conf.LLM_MODEL,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "detail": str(e)}


@app.get("/api/sources")
async def sources():
    labels = {
        "ai": "AI",
        "java": "Java",
        "test": "软件测试",
        "ops": "运维",
        "bigdata": "大数据",
    }
    return {
        "sources": [
            {"value": source, "label": labels.get(source, source.upper())}
            for source in conf.VALID_SOURCES
        ]
    }


@app.get("/api/project")
async def project_info():
    return {
        "ragas": load_ragas_scores(),
        "subjects": conf.VALID_SOURCES,
        "models": {
            "llm": conf.LLM_MODEL,
            "embedding": "BGE-M3 dense + sparse",
            "reranker": "BGE-Reranker-Large",
            "intent": "BERT binary classifier",
        },
        "retrieval": {
            "parent_chunk_size": conf.PARENT_CHUNK_SIZE,
            "child_chunk_size": conf.CHILD_CHUNK_SIZE,
            "chunk_overlap": conf.CHUNK_OVERLAP,
            "retrieval_k": conf.RETRIEVAL_K,
            "candidate_m": conf.CANDIDATE_M,
            "ranker": "Milvus WeightedRanker sparse=0.7 dense=1.0",
        },
        "storage": {
            "mysql": conf.MYSQL_DATABASE,
            "redis": f"{conf.REDIS_HOST}:{conf.REDIS_PORT}/{conf.REDIS_DB}",
            "milvus": f"{conf.MILVUS_HOST}:{conf.MILVUS_PORT}/{conf.MILVUS_DATABASE_NAME}.{conf.MILVUS_COLLECTION_NAME}",
        },
    }


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = body.get("query", "")
    source_filter = body.get("source_filter")
    session_id = body.get("session_id")
    history = body.get("history", [])

    async def event_stream():
        try:
            if not (query or "").strip():
                yield sse({"type": "error", "error": "empty_query", "content": "请输入问题"})
                return
            qa = get_qa_system()
            for event in qa.stream_answer(
                query,
                source_filter=source_filter,
                session_id=session_id,
                history=history,
            ):
                yield sse(event)
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield sse({"type": "error", "error": str(e), "content": "服务暂时不可用，请检查后端依赖。"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
