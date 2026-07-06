from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    user_id: str = Field(default="default_user", description="用户 ID")
    top_k: int | None = Field(default=None, ge=1, le=20, description="检索条数")
    prefer_graph: bool = Field(default=True, description="是否优先使用 Neo4j 图谱模式")


class RetrievedDocument(BaseModel):
    id: str
    name: str
    type: str
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["graph_rag", "local_rag", "llm_only"]
    sources: list[RetrievedDocument] = Field(default_factory=list)
    graph_available: bool = False
    safety_notice: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: list[RetrievedDocument]


class HealthResponse(BaseModel):
    status: str
    project: str
    model: str
    model_api_configured: bool
    index_exists: bool
    documents_exists: bool


class GraphStatusResponse(BaseModel):
    available: bool
    message: str
    node_count: int | None = None
    relationship_count: int | None = None
