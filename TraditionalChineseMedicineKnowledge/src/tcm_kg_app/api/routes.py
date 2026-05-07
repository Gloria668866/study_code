from fastapi import APIRouter

from tcm_kg_app.config import get_settings
from tcm_kg_app.core.qa_service import QaService
from tcm_kg_app.graph.neo4j_client import Neo4jService
from tcm_kg_app.rag.vector_store import VectorStore
from tcm_kg_app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    GraphStatusResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/api")
qa_service = QaService()
vector_store = VectorStore()
graph_service = Neo4jService()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        project="TraditionalChineseMedicineKnowledge",
        model=settings.model_name,
        model_api_configured=bool(settings.model_api_key),
        index_exists=settings.faiss_index_path.exists() and settings.faiss_meta_path.exists(),
        documents_exists=settings.documents_path.exists(),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    top_k = request.top_k or settings.top_k
    return qa_service.answer(request.question, top_k=top_k, prefer_graph=request.prefer_graph)


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    return SearchResponse(results=vector_store.search(request.query, top_k=request.top_k))


@router.get("/graph/status", response_model=GraphStatusResponse)
def graph_status() -> GraphStatusResponse:
    return graph_service.status()
