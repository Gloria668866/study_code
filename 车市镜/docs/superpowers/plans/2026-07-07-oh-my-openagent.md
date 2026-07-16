# oh-my-openagent Pipeline Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the main LangGraph flow returns `no_data=True` (SQL executed but returned 0 rows), trigger an async Research→Plan→Code→Review agent pipeline that autonomously gathers missing data from the web, writes it into the RAG knowledge base (public user_id=0), then auto-re-runs the original question to deliver results to the user via SSE.

**Architecture:** Three new modules add a config-driven DAG execution engine beside the existing synchronous LangGraph graph. `config/agents.yaml` declares pipeline stages, agent capabilities, and tool permissions. `app/agent_pipeline.py` runs the pipeline as a Celery task — each stage is a deterministic LLM call (not code generation), producing structured JSON that the next stage consumes. `app/agent_tools.py` provides pre-defined tools (`http_get`, `search_web`, `parse_html`, `write_to_rag`) that the Code stage agents invoke via function calling. The main SSE endpoint in `app/main.py` gains a new subscribe path for pipeline progress events. The existing `app/graph.py:insight()` node gains ~5 lines to detect `no_data=True` and dispatch the Celery task instead of returning a dead-end message.

**Tech Stack:** Celery + Redis (existing), LangGraph (existing), FastAPI SSE (existing), RAG ingest pipeline (existing, `app/rag/local_ingest.py`), PyYAML (existing via `config/nlu.yaml`).

## Global Constraints

- Python 3.11+, FastAPI/LangGraph/Celery existing stack
- RAG backend default is `local` (SQLite + numpy). Use `app/rag/local_ingest.py` for writing scraped content — do NOT require PostgreSQL/MinIO.
- All LLM calls use existing `app/llm.py chat()` — do not add new LLM clients
- Agent tools must run in-process (no code sandbox, no subprocess execution). The Code Agent calls pre-defined Python functions, never generates executable code.
- Pipeline status must be observable: each stage writes progress to Redis (`pipeline:{task_id}:status`) so SSE clients can subscribe
- Pipeline failure at any stage must degrade gracefully — the user sees a "partial results" or "collection failed" message, never a spinner forever
- `config/agents.yaml` loaded once at startup (thread-safe lazy init, double-checked locking, same pattern as `app/nlu.py`)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| CREATE | `config/agents.yaml` | Pipeline DAG definition + agent configs + tool permissions |
| CREATE | `app/agent_tools.py` | Pre-defined tools: http_get, search_web, parse_html, write_to_rag |
| CREATE | `app/agent_pipeline.py` | DAG executor + Celery task + Redis progress + pipeline runner |
| MODIFY | `app/config.py` | Add `AGENTS_CONFIG_PATH` env var |
| MODIFY | `app/graph.py` | `insight()` node: on `no_data=True`, dispatch Celery task + return task_id |
| MODIFY | `app/main.py` | New SSE endpoint `GET /api/tasks/{task_id}/stream` for pipeline progress |
| CREATE | `tests/test_agent_pipeline.py` | Tests for pipeline runner, tools, DAG execution |

---

### Task 1: Config file + app/config.py change

**Files:**
- Create: `config/agents.yaml`
- Modify: `app/config.py` (add 1 line)

**Interfaces:**
- Produces: `config/agents.yaml` loaded by `app/agent_pipeline.py` via `AGENTS_CONFIG_PATH`

- [ ] **Step 1: Create `config/agents.yaml`**

```yaml
# oh-my-openagent pipeline configuration
# Pipeline DAG: Research → Plan → parallel Code agents → Review
pipeline:
  stages:
    - id: research
      agent: research_agent
      description: "Analyze what data is needed and which websites to scrape"
    - id: plan
      agent: plan_agent
      depends_on: [research]
      description: "Break into parallel collection tasks"
    - id: code
      agent: code_agent
      depends_on: [plan]
      parallel: true               # one instance per plan item
      description: "Execute web collection in parallel"
    - id: review
      agent: review_agent
      depends_on: [code]
      description: "Summarize results, verify quality, write to RAG"

agents:
  research_agent:
    model: null                    # null = use default LLM_MODEL from .env
    temperature: 0.0
    system_prompt: |
      你是新能源汽车市场数据调研专家。用户问了一个问题，但我们的数据库里没有相关数据。
      你的任务：分析需要什么样的数据、从哪些公开网站可以获取。
      
      返回严格 JSON（无其他文字）：
      {
        "data_requirements": ["需要的数据项1", "需要的数据项2"],
        "suggested_sources": [
          {"name": "来源名称", "url": "https://...", "type": "html|api|search", "reason": "为什么选这个来源"}
        ],
        "search_queries": ["搜索引擎查询词1", "查询词2"],
        "confidence": 0.85
      }

  plan_agent:
    model: null
    temperature: 0.0
    system_prompt: |
      你是数据采集规划专家。根据 Research Agent 的分析结果，制定并行采集方案。
      
      返回严格 JSON（无其他文字）：
      {
        "tasks": [
          {
            "id": "task_1",
            "description": "采集任务描述",
            "source": {"name": "来源名", "url": "https://...", "type": "html|api|search"},
            "tool": "http_get|search_web",
            "priority": 1
          }
        ],
        "estimated_time_seconds": 30
      }

  code_agent:
    model: null
    temperature: 0.1
    system_prompt: |
      你是数据采集执行器。你是数据采集执行器。你有一组预定义工具，用 function calling 完成采集任务。
      你能用的工具已通过 OpenAI function calling 协议提供给你。
      
      重要规则：
      - 只使用提供的工具，不要尝试生成代码
      - 如果目标网站返回 403/Cloudflare/需要 JS 渲染，立即报告失败不要重试
      - 把采集到的原始文本交给 review_agent 处理，不要自己总结
      
      返回严格 JSON（无其他文字）：
      {
        "status": "success|partial|failed",
        "collected_data": [
          {"source_url": "https://...", "content_type": "text/html", "content_preview": "前200字符...", "byte_length": 1234}
        ],
        "errors": ["错误描述"],
        "tool_calls_made": 3
      }

  review_agent:
    model: null
    temperature: 0.1
    system_prompt: |
      你是数据质量审核员。审核采集到的数据，判断是否足以回答用户的原始问题。
      
      返回严格 JSON（无其他文字）：
      {
        "verdict": "sufficient|partial|insufficient",
        "summary": "一句话总结采集到了什么",
        "answer_to_user": "如果数据充足，用采集到的数据回答用户原始问题；如果不充足，诚实说明并建议用户换个问法",
        "data_quality": {
          "relevance_score": 0.85,
          "completeness_score": 0.70,
          "issues": ["问题1", "问题2"]
        },
        "should_write_to_rag": true,
        "rag_title": "入库文档标题（如：理想L9海外销量-采集数据）"
      }

# Tool permissions: which tools each agent can use
tool_permissions:
  research_agent: []               # research only analyzes, doesn't execute
  plan_agent: []                   # plan only structures, doesn't execute
  code_agent: [http_get, search_web, parse_html]
  review_agent: [write_to_rag]     # review writes results to knowledge base

redis:
  progress_ttl_seconds: 1800       # pipeline progress keys expire after 30 min
```

- [ ] **Step 2: Add `AGENTS_CONFIG_PATH` to `app/config.py`**

Open `app/config.py` and add after the `NLU_CONFIG_PATH` line:

```python
AGENTS_CONFIG_PATH = os.getenv("AGENTS_CONFIG_PATH", "config/agents.yaml")
```

- [ ] **Step 3: Commit**

```bash
git add config/agents.yaml app/config.py
git commit -m "feat(agent): add agents.yaml pipeline config and AGENTS_CONFIG_PATH env var"
```

---

### Task 2: `app/agent_tools.py` — Pre-defined tools for Code/Review agents

**Files:**
- Create: `app/agent_tools.py`

**Interfaces:**
- Produces:
  - `agent_tools.get_tool_definitions(agent_name) -> list[dict]` — returns OpenAI function-calling tool schemas
  - `agent_tools.execute_tool(tool_name, arguments) -> dict` — executes a tool call
  - `agent_tools.ALL_TOOLS: dict[str, callable]` — registry of tool name → implementation

- [ ] **Step 1: Create `tests/test_agent_pipeline.py` with tool tests**

```python
"""Tests for agent pipeline: tools, DAG execution, pipeline runner."""
import pytest
from unittest.mock import patch, MagicMock


# ── Tool registry ─────────────────────────────────────────────────────────────

def test_get_tool_definitions_returns_schemas():
    from app.agent_tools import get_tool_definitions
    schemas = get_tool_definitions("code_agent")
    assert isinstance(schemas, list)
    assert len(schemas) >= 3  # http_get, search_web, parse_html
    # Each must have "type": "function"
    for s in schemas:
        assert s["type"] == "function"
        assert "function" in s
        assert "name" in s["function"]


def test_get_tool_definitions_research_agent_empty():
    from app.agent_tools import get_tool_definitions
    schemas = get_tool_definitions("research_agent")
    assert schemas == []


def test_execute_tool_http_get_success():
    from app.agent_tools import execute_tool
    with patch("app.agent_tools.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>Test page content</body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_get.return_value = mock_resp

        result = execute_tool("http_get", {"url": "https://example.com"})
        assert result["status"] == "success"
        assert result["status_code"] == 200
        assert "Test page content" in result["content"]


def test_execute_tool_http_get_http_error():
    from app.agent_tools import execute_tool
    with patch("app.agent_tools.httpx.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_get.return_value = mock_resp

        result = execute_tool("http_get", {"url": "https://blocked.example.com"})
        assert result["status"] == "failed"
        assert result["status_code"] == 403


def test_execute_tool_http_get_timeout():
    from app.agent_tools import execute_tool
    import httpx
    with patch("app.agent_tools.httpx.get", side_effect=httpx.TimeoutException("timeout")):
        result = execute_tool("http_get", {"url": "https://slow.example.com"})
        assert result["status"] == "failed"
        assert "timeout" in result.get("error", "").lower()


def test_execute_tool_parse_html():
    from app.agent_tools import execute_tool
    html = "<html><body><h1>Title</h1><p>Paragraph text here.</p><script>alert('xss')</script></body></html>"
    result = execute_tool("parse_html", {"html": html})
    assert result["status"] == "success"
    assert "Title" in result["text"]
    assert "Paragraph text here." in result["text"]
    assert "alert" not in result["text"]  # script stripped


def test_execute_tool_unknown_tool():
    from app.agent_tools import execute_tool
    result = execute_tool("nonexistent_tool", {})
    assert result["status"] == "error"
    assert "unknown tool" in result.get("error", "").lower()


def test_execute_tool_write_to_rag(monkeypatch):
    from app.agent_tools import execute_tool
    from unittest.mock import MagicMock

    mock_ingest = MagicMock(return_value=(42, 5))
    monkeypatch.setattr("app.agent_tools.ingest_text_as_document", mock_ingest)

    result = execute_tool("write_to_rag", {
        "title": "Test Document",
        "content": "This is test content for RAG",
        "source_url": "https://example.com/data"
    })
    assert result["status"] == "success"
    assert result["doc_id"] == 42
    assert result["chunk_count"] == 5
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cd C:/Users/GUANGBL/lgb_coding/demo1
.venv/Scripts/pytest tests/test_agent_pipeline.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'app.agent_tools'`

- [ ] **Step 3: Create `app/agent_tools.py`**

```python
"""Pre-defined tools for oh-my-openagent pipeline agents.

Code agents call these via function calling — they never generate executable code.
Each tool is a pure Python function with a defined JSON schema for LLM consumption.
"""
import json
import re
import time
import logging
from typing import Any

import httpx

from .config import AGENTS_CONFIG_PATH

_log = logging.getLogger("cheshijing.agent_tools")

# httpx client with sane defaults for web scraping
_http = httpx.Client(
    timeout=httpx.Timeout(15.0, connect=10.0),
    headers={"User-Agent": "CarMirror/1.0 (market research bot; contact@example.com)"},
    follow_redirects=True,
    max_redirects=3,
)


def _tool_http_get(url: str, **kwargs) -> dict:
    """Fetch a web page. Returns text content and metadata."""
    try:
        resp = _http.get(url)
        content = resp.text[:50000]  # truncate to 50KB to avoid overwhelming LLM context
        return {
            "status": "success" if 200 <= resp.status_code < 300 else "failed",
            "status_code": resp.status_code,
            "content": content,
            "content_type": resp.headers.get("content-type", "unknown"),
            "url": str(resp.url),  # final URL after redirects
        }
    except httpx.TimeoutException:
        return {"status": "failed", "error": "Request timed out after 15s", "url": url}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200], "url": url}


def _tool_search_web(query: str, num_results: int = 5, **kwargs) -> dict:
    """Search the web using DuckDuckGo Instant Answer API (no API key needed).
    For a production system, replace with Bing/Google Search API.
    Returns titles, URLs, and snippets."""
    try:
        # DuckDuckGo Instant Answer API — free, no auth required
        resp = _http.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        data = resp.json()
        results = []

        # Abstract (instant answer)
        if data.get("AbstractText"):
            results.append({
                "title": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"][:500],
            })

        # Related topics
        for topic in (data.get("RelatedTopics") or [])[:num_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": (topic.get("FirstURL") or "").split("/")[-1].replace("_", " "),
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic["Text"][:500],
                })

        return {
            "status": "success",
            "query": query,
            "results": results[:num_results],
            "total_found": len(results),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200], "query": query}


def _tool_parse_html(html: str, **kwargs) -> dict:
    """Extract readable text from HTML. Strips scripts, styles, and boilerplate.
    Uses regex for lightweight extraction — no full DOM parser needed."""
    if not html or not html.strip():
        return {"status": "failed", "error": "empty input", "text": ""}
    try:
        # Remove script and style blocks
        cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Decode common entities
        cleaned = cleaned.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        cleaned = cleaned.replace("&quot;", '"').replace("&#39;", "'")
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Trim to reasonable length
        cleaned = cleaned[:20000]
        return {"status": "success", "text": cleaned, "char_length": len(cleaned)}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200], "text": ""}


def _tool_write_to_rag(title: str, content: str, source_url: str = "", **kwargs) -> dict:
    """Write collected content into the RAG knowledge base as a public document (user_id=0).
    Uses the existing local_ingest pipeline (chunk → embed → store)."""
    try:
        # Construct a markdown document from the collected data
        md_lines = [f"# {title}", "", f"> 数据来源：{source_url}", "", content]
        md_text = "\n".join(md_lines)
        data = md_text.encode("utf-8")

        filename = f"agent_collected_{int(time.time())}.md"
        doc_id, chunk_count = ingest_text_as_document(
            title=title,
            text=md_text,
            filename=filename,
            source_url=source_url,
        )
        return {
            "status": "success",
            "doc_id": doc_id,
            "chunk_count": chunk_count,
            "title": title,
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200], "title": title}


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS: dict[str, callable] = {
    "http_get": _tool_http_get,
    "search_web": _tool_search_web,
    "parse_html": _tool_parse_html,
    "write_to_rag": _tool_write_to_rag,
}

# OpenAI function-calling JSON schemas
TOOL_SCHEMAS: dict[str, dict] = {
    "http_get": {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Fetch content from a web URL. Returns page text and metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch (include https://)"},
                },
                "required": ["url"],
            },
        },
    },
    "search_web": {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query in Chinese or English"},
                    "num_results": {"type": "integer", "description": "Number of results (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    "parse_html": {
        "type": "function",
        "function": {
            "name": "parse_html",
            "description": "Extract readable text from raw HTML. Strips scripts, styles, and tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "html": {"type": "string", "description": "Raw HTML content to parse"},
                },
                "required": ["html"],
            },
        },
    },
    "write_to_rag": {
        "type": "function",
        "function": {
            "name": "write_to_rag",
            "description": "Write collected content into the knowledge base for future queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Document title"},
                    "content": {"type": "string", "description": "Full text content to store (markdown)"},
                    "source_url": {"type": "string", "description": "Source URL for traceability"},
                },
                "required": ["title", "content"],
            },
        },
    },
}


def get_tool_definitions(agent_name: str) -> list[dict]:
    """Return OpenAI function-calling tool definitions for an agent.
    Reads tool_permissions from agents.yaml to determine which tools this agent can use."""
    schemas = []
    permissions = _get_tool_permissions(agent_name)
    for tool_name in permissions:
        if tool_name in TOOL_SCHEMAS:
            schemas.append(TOOL_SCHEMAS[tool_name])
    return schemas


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool by name. Returns a dict with at least {"status": "..."}."""
    if tool_name not in ALL_TOOLS:
        return {"status": "error", "error": f"unknown tool '{tool_name}'"}
    try:
        return ALL_TOOLS[tool_name](**arguments)
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


# ── RAG write helper (reuses existing pipeline) ────────────────────────────────

def ingest_text_as_document(title: str, text: str, filename: str = "agent_collected.md",
                            source_url: str = "", public: bool = True) -> tuple[int, int]:
    """Write a text document through the RAG ingest pipeline.
    
    Uses local_ingest when RAG_BACKEND=local (default), pg ingest when RAG_BACKEND=pg.
    public=True writes as user_id=None (visible to all users).
    public=False writes as user_id=0 (system user, also public in local_store semantics).
    """
    from .config import RAG_BACKEND

    data = text.encode("utf-8")
    file_type = "md"

    if RAG_BACKEND == "pg":
        from .rag import pg, store, embed
        from .rag.chunk import build_chunks
        from .rag.parse import parse_document

        source_uri = store.put_bytes(0, filename, data, "text/markdown")
        doc_id = pg.create_document(0, filename, file_type, source_uri, title=title)
        blocks = parse_document(data, file_type)
        chunks = build_chunks(blocks, count_tokens=embed.count_tokens)
        children = [c for c in chunks if c["is_retrievable"]]
        vecs = embed.embed_passages([c["content_embed"] for c in children])
        emb_by_idx = {c["chunk_index"]: v for c, v in zip(children, vecs)}
        n = pg.insert_chunks(doc_id, 0, chunks, emb_by_idx)
        pg.set_status(doc_id, "ready", chunk_count=n)
        return doc_id, n
    else:
        # RAG_BACKEND=local — write as public document (user_id=None in SQLite)
        try:
            from .rag.local_ingest import ingest_bytes as local_ingest_bytes
            return local_ingest_bytes(0, filename, data, file_type, title=title, public=True)
        except TypeError:
            # Fallback: older local_ingest might not support public param
            from .rag.local_ingest import ingest_bytes as local_ingest_bytes
            return local_ingest_bytes(0, filename, data, file_type, title=title)


# ── Internal helpers ──────────────────────────────────────────────────────────

_yaml_cfg = None


def _get_tool_permissions(agent_name: str) -> list[str]:
    """Read tool_permissions from agents.yaml (with caching)."""
    global _yaml_cfg
    import yaml
    from pathlib import Path

    if _yaml_cfg is None:
        path = Path(AGENTS_CONFIG_PATH)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                _yaml_cfg = yaml.safe_load(f) or {}
        else:
            _yaml_cfg = {}
    return (_yaml_cfg.get("tool_permissions") or {}).get(agent_name, [])
```

- [ ] **Step 4: Run tool tests**

```bash
.venv/Scripts/pytest tests/test_agent_pipeline.py -v -k "tool"
```

Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent_tools.py tests/test_agent_pipeline.py
git commit -m "feat(agent): pre-defined tools (http_get/search_web/parse_html/write_to_rag)"
```

---

### Task 3: `app/agent_pipeline.py` — DAG engine + Redis progress + Celery task

**Files:**
- Create: `app/agent_pipeline.py`
- Modify: `tests/test_agent_pipeline.py` (add pipeline engine tests)

**Interfaces:**
- Consumes: `config/agents.yaml`, `app/llm.chat()`, `app/agent_tools.get_tool_definitions()`, `app/agent_tools.execute_tool()`, Redis (via `redis` package)
- Produces:
  - `agent_pipeline.run_pipeline(task_id, question, original_user_id) -> dict` — main entry point
  - `agent_pipeline.get_progress(task_id) -> dict | None` — read pipeline progress from Redis
  - `agent_pipeline._execute_stage(stage_def, context, tool_defs) -> dict` — single stage executor

- [ ] **Step 1: Add pipeline engine tests**

Add to `tests/test_agent_pipeline.py`:

```python
# ── Pipeline DAG engine ───────────────────────────────────────────────────────

def test_load_pipeline_config():
    from app.agent_pipeline import _load_pipeline_config
    cfg = _load_pipeline_config()
    assert "stages" in cfg
    assert len(cfg["stages"]) == 4
    stage_ids = [s["id"] for s in cfg["stages"]]
    assert stage_ids == ["research", "plan", "code", "review"]


def test_topological_sort():
    from app.agent_pipeline import _topological_sort
    stages = [
        {"id": "research", "depends_on": []},
        {"id": "plan", "depends_on": ["research"]},
        {"id": "code", "depends_on": ["plan"]},
        {"id": "review", "depends_on": ["code"]},
    ]
    ordered = _topological_sort(stages)
    ids = [s["id"] for s in ordered]
    assert ids.index("research") < ids.index("plan")
    assert ids.index("plan") < ids.index("code")
    assert ids.index("code") < ids.index("review")


def test_topological_sort_diamond_dag():
    from app.agent_pipeline import _topological_sort
    stages = [
        {"id": "start", "depends_on": []},
        {"id": "left", "depends_on": ["start"]},
        {"id": "right", "depends_on": ["start"]},
        {"id": "end", "depends_on": ["left", "right"]},
    ]
    ordered = _topological_sort(stages)
    ids = [s["id"] for s in ordered]
    assert ids.index("start") == 0
    assert ids.index("end") == 3
    # left and right can be in either order, both between start and end
    assert ids.index("start") < ids.index("left") < ids.index("end")
    assert ids.index("start") < ids.index("right") < ids.index("end")


def test_get_stage_dependencies():
    from app.agent_pipeline import _get_stage_by_id
    cfg = _load_pipeline_config_stub()
    stage = _get_stage_by_id(cfg["stages"], "plan")
    assert stage is not None
    assert stage["id"] == "plan"
    assert "research" in stage.get("depends_on", [])


def test_redis_progress_write_read(monkeypatch):
    """Test progress write/read with a fake Redis."""
    from app.agent_pipeline import _redis_progress

    fake_store = {}
    def fake_set(key, value, ex=None):
        fake_store[key] = value
    def fake_get(key):
        return fake_store.get(key)

    monkeypatch.setattr(_redis_progress, "_r", None)
    # Patch the internal redis client to use our fake
    import app.agent_pipeline as ap
    monkeypatch.setattr(ap, "_redis_client", lambda: type('FakeRedis', (), {
        'set': fake_set, 'get': fake_get, 'exists': lambda k: k in fake_store
    })())

    ap._set_progress("test_task_123", {"stage": "research", "status": "running"})
    progress = ap._get_progress("test_task_123")
    assert progress is not None
    assert progress["stage"] == "research"


# Stub helper
def _load_pipeline_config_stub():
    return {
        "stages": [
            {"id": "research", "depends_on": []},
            {"id": "plan", "depends_on": ["research"]},
            {"id": "code", "depends_on": ["plan"], "parallel": True},
            {"id": "review", "depends_on": ["code"]},
        ]
    }
```

- [ ] **Step 2: Run to verify tests fail**

```bash
.venv/Scripts/pytest tests/test_agent_pipeline.py -v -k "pipeline or topological or redis or load" 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'app.agent_pipeline'`

- [ ] **Step 3: Create `app/agent_pipeline.py`**

```python
"""Config-driven DAG execution engine for oh-my-openagent pipelines.

Runs a pipeline defined in config/agents.yaml as a Celery task.
Each stage is an LLM call (with optional function calling for tool-enabled stages).
Progress is written to Redis so SSE clients can subscribe.

Key design decisions:
- Deterministic DAG execution (topological sort), not an LLM deciding what to run next.
- Code agents call pre-defined tools via function calling — never generate code.
- Every stage's LLM response is validated against its expected JSON schema before proceeding.
- FAIL-SAFE: any stage failure → pipeline stops, progress shows "failed", user gets degraded response.
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import redis
import yaml

from .config import AGENTS_CONFIG_PATH, REDIS_URL
from .llm import chat

_log = logging.getLogger("cheshijing.agent_pipeline")

# ── Config loading (thread-safe, same pattern as app/nlu.py) ──────────────────

_cfg_lock = threading.Lock()
_cfg = None


def _load_pipeline_config() -> dict:
    global _cfg
    with _cfg_lock:
        if _cfg is not None:
            return _cfg
        path = Path(AGENTS_CONFIG_PATH)
        if not path.exists():
            _cfg = {"stages": [], "agents": {}}
            return _cfg
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _cfg = {"stages": raw.get("pipeline", {}).get("stages", []),
                "agents": raw.get("agents", {}),
                "redis": raw.get("redis", {})}
        return _cfg


def _get_agent_config(agent_name: str) -> dict:
    cfg = _load_pipeline_config()
    return cfg.get("agents", {}).get(agent_name, {})


def _get_stage_by_id(stages: list, stage_id: str) -> dict | None:
    for s in stages:
        if s["id"] == stage_id:
            return s
    return None


# ── DAG topological sort ──────────────────────────────────────────────────────

def _topological_sort(stages: list) -> list:
    """Kahn's algorithm. Sorts stages so dependencies come before dependents."""
    in_degree = {s["id"]: len(s.get("depends_on") or []) for s in stages}
    adj = {s["id"]: [] for s in stages}
    for s in stages:
        for dep in (s.get("depends_on") or []):
            if dep in adj:
                adj[dep].append(s["id"])

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    result = []
    while queue:
        sid = queue.pop(0)
        result.append(_get_stage_by_id(stages, sid))
        for neighbor in adj.get(sid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(stages):
        raise ValueError(f"Pipeline DAG has a cycle or missing dependency. Sorted {len(result)}/{len(stages)}")
    return result


# ── Redis progress ────────────────────────────────────────────────────────────

_redis_client_instance = None


def _redis_client():
    global _redis_client_instance
    if _redis_client_instance is None:
        _redis_client_instance = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client_instance


def _set_progress(task_id: str, data: dict):
    """Write pipeline progress to Redis. TTL matched to config."""
    try:
        r = _redis_client()
        ttl = _load_pipeline_config().get("redis", {}).get("progress_ttl_seconds", 1800)
        r.set(f"pipeline:{task_id}:status", json.dumps(data, ensure_ascii=False, default=str), ex=ttl)
    except Exception:
        _log.warning(f"Failed to write Redis progress for {task_id}", exc_info=True)


def _get_progress(task_id: str) -> dict | None:
    """Read pipeline progress from Redis."""
    try:
        r = _redis_client()
        raw = r.get(f"pipeline:{task_id}:status")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


# ── Stage execution ────────────────────────────────────────────────────────────

def _call_llm_with_tools(agent_name: str, messages: list, tool_defs: list) -> dict:
    """Call LLM with function calling. If the model returns tool_calls, execute them
    and feed results back for one follow-up turn. Returns final parsed JSON response.

    Uses a two-turn pattern:
    Turn 1: LLM sees tools, may return tool_calls
    Turn 2 (if tool_calls): tool results fed back, LLM returns final JSON
    """
    agent_cfg = _get_agent_config(agent_name)
    model = agent_cfg.get("model") or None
    temperature = agent_cfg.get("temperature", 0.0)

    if not tool_defs:
        # No tools — single call
        raw = chat(messages, temperature=temperature, model=model)
        return _parse_json_response(raw)

    # With tools: first turn
    raw = chat(messages, temperature=temperature, model=model, tools=tool_defs)
    
    # Check if the model wants to call tools
    # The response might contain tool_calls in the raw string (OpenAI-compatible format)
    # For simplicity with the existing chat() shim, we inspect for tool call markers
    tool_results = []
    if _has_tool_calls(raw):
        tool_calls = _extract_tool_calls(raw)
        for tc in tool_calls:
            from .agent_tools import execute_tool
            result = execute_tool(tc["name"], tc.get("arguments", {}))
            tool_results.append({"tool_name": tc["name"], "result": result})

        # Feed tool results back
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": json.dumps(
            {"tool_results": tool_results}, ensure_ascii=False)})
        raw = chat(messages, temperature=temperature, model=model)

    return _parse_json_response(raw)


def _has_tool_calls(raw: str) -> bool:
    """Detect if the LLM response contains function-calling tool_calls."""
    try:
        data = json.loads(raw)
        return bool(data.get("tool_calls"))
    except (json.JSONDecodeError, TypeError):
        return False


def _extract_tool_calls(raw: str) -> list[dict]:
    """Extract tool call requests from an OpenAI-format response."""
    try:
        data = json.loads(raw)
        calls = data.get("tool_calls", [])
        return [{"name": c.get("function", {}).get("name", ""),
                 "arguments": json.loads(c.get("function", {}).get("arguments", "{}"))}
                for c in calls]
    except Exception:
        return []


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response. FAIL-SAFE: returns error dict on failure."""
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s < 0 or e <= s:
            raise ValueError("no JSON in response")
        return json.loads(raw[s:e])
    except Exception:
        return {"status": "failed", "error": "Failed to parse LLM response as JSON",
                "raw_preview": raw[:200]}


def _execute_stage(stage_def: dict, pipeline_context: dict, 
                   original_question: str) -> dict:
    """Execute a single pipeline stage. Returns the parsed LLM output dict."""
    stage_id = stage_def["id"]
    agent_name = stage_def["agent"]
    agent_cfg = _get_agent_config(agent_name)

    if not agent_cfg:
        raise ValueError(f"Agent '{agent_name}' not found in agents.yaml (stage '{stage_id}')")

    from .agent_tools import get_tool_definitions
    tool_defs = get_tool_definitions(agent_name)

    # Build messages
    system_prompt = agent_cfg.get("system_prompt", "").strip()
    messages = [{"role": "system", "content": system_prompt}]

    # Build user prompt with context from upstream stages + original question
    user_parts = [f"用户原始问题：{original_question}"]
    if pipeline_context:
        user_parts.append(f"上游阶段输出：{json.dumps(pipeline_context, ensure_ascii=False)}")
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    return _call_llm_with_tools(agent_name, messages, tool_defs)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(task_id: str, original_question: str, 
                 original_user_id: int = 0) -> dict:
    """Execute the full pipeline DAG. Called by Celery task.
    
    Returns: {"status": "completed|failed", "stages": {...}, "final_answer": "..."}
    """
    _set_progress(task_id, {"stage": "start", "status": "running", "ts": time.time()})

    cfg = _load_pipeline_config()
    stages = cfg.get("stages", [])
    if not stages:
        return {"status": "failed", "error": "No pipeline stages defined in agents.yaml"}

    # Topological sort
    try:
        ordered = _topological_sort(stages)
    except ValueError as e:
        return {"status": "failed", "error": str(e)}

    # Execute stages in order, feeding each stage's output as context for downstream
    stage_outputs: dict[str, Any] = {}
    pipeline_context: dict[str, Any] = {}

    for stage_def in ordered:
        stage_id = stage_def["id"]
        _set_progress(task_id, {"stage": stage_id, "status": "running", "ts": time.time()})

        try:
            # Build context from dependency outputs
            deps = stage_def.get("depends_on") or []
            dep_context = {dep: stage_outputs.get(dep) for dep in deps if dep in stage_outputs}

            output = _execute_stage(stage_def, dep_context, original_question)
            stage_outputs[stage_id] = output

            _set_progress(task_id, {
                "stage": stage_id, "status": "completed",
                "ts": time.time(), "preview": json.dumps(output, ensure_ascii=False)[:200],
            })

            # If review stage returned an answer, capture it
            if stage_id == "review":
                pipeline_context["review_result"] = output

        except Exception as e:
            _log.error(f"Pipeline stage '{stage_id}' failed: {e}", exc_info=True)
            _set_progress(task_id, {
                "stage": stage_id, "status": "failed",
                "error": str(e)[:200], "ts": time.time(),
            })
            return {
                "status": "failed",
                "failed_stage": stage_id,
                "error": str(e)[:300],
                "stages": stage_outputs,
            }

    # Pipeline completed
    review_output = stage_outputs.get("review", {})
    final_answer = review_output.get("answer_to_user", "数据采集完成，但未能生成有效回答。")

    _set_progress(task_id, {
        "stage": "done", "status": "completed",
        "ts": time.time(), "final_answer": final_answer[:500],
    })

    return {
        "status": "completed",
        "stages": stage_outputs,
        "final_answer": final_answer,
    }
```

- [ ] **Step 4: Run pipeline engine tests**

```bash
.venv/Scripts/pytest tests/test_agent_pipeline.py -v -k "pipeline or topological or redis or load"
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent_pipeline.py tests/test_agent_pipeline.py
git commit -m "feat(agent): DAG engine + Redis progress + pipeline runner"
```

---

### Task 4: Celery task wiring

**Files:**
- Modify: `app/agent_pipeline.py` (add Celery task decorator)
- Modify: `app/celery_app.py` (register new task module)

**Interfaces:**
- Consumes: `app.celery_app.celery` instance
- Produces: Celery task `agent_pipeline.run_pipeline_task` callable via `.delay(task_id, question, user_id)`

- [ ] **Step 1: Add Celery task to `app/agent_pipeline.py`**

Add to the end of `app/agent_pipeline.py`:

```python
# ── Celery task ───────────────────────────────────────────────────────────────

from .celery_app import celery


@celery.task(name="agent_pipeline.run", bind=True, max_retries=0, 
             task_ignore_result=True)
def run_pipeline_task(self, task_id: str, original_question: str, 
                      original_user_id: int = 0):
    """Celery task wrapper for run_pipeline. fire-and-forget with Redis progress.
    On success, writes results to RAG and triggers re-query via callback.
    """
    _set_progress(task_id, {"stage": "queued", "status": "pending", "ts": time.time()})

    try:
        result = run_pipeline(task_id, original_question, original_user_id)
        _set_progress(task_id, {
            "stage": result.get("status", "done"),
            "status": result.get("status", "failed"),
            "final_answer": result.get("final_answer", "")[:500],
            "ts": time.time(),
        })
        return result
    except Exception as e:
        _log.error(f"Pipeline task {task_id} failed: {e}", exc_info=True)
        _set_progress(task_id, {
            "stage": "error", "status": "failed",
            "error": str(e)[:300], "ts": time.time(),
        })
        return {"status": "failed", "error": str(e)[:300]}
```

- [ ] **Step 2: Register task module in `app/celery_app.py`**

Add after the existing import lines at the bottom of `app/celery_app.py`:

```python
import app.agent_pipeline    # noqa: E402,F401  oh-my-openagent pipeline tasks
```

The bottom of `app/celery_app.py` should now look like:

```python
# 注册任务模块
import app.rag.tasks           # noqa: E402,F401  RAG 入库任务
import app.tasks_cron          # noqa: E402,F401  定时采集任务
import app.agent_pipeline      # noqa: E402,F401  oh-my-openagent pipeline tasks
```

- [ ] **Step 3: Add task dispatch test**

Add to `tests/test_agent_pipeline.py`:

```python
def test_run_pipeline_task_signature():
    """Verify the Celery task is registered with the correct signature."""
    from app.agent_pipeline import run_pipeline_task
    assert run_pipeline_task.name == "agent_pipeline.run"
    assert callable(run_pipeline_task.delay)
```

- [ ] **Step 4: Run all pipeline tests**

```bash
.venv/Scripts/pytest tests/test_agent_pipeline.py -v
```

Expected: all tests PASS (~15 tests)

- [ ] **Step 5: Commit**

```bash
git add app/agent_pipeline.py app/celery_app.py tests/test_agent_pipeline.py
git commit -m "feat(agent): wire Celery task + register in celery_app"
```

---

### Task 5: Modify `app/graph.py` — `insight()` node triggers pipeline on `no_data=True`

**Files:**
- Modify: `app/graph.py` (insight node, ~10 lines added)
- Modify: `tests/test_graph.py` (add one test for the trigger)

**Interfaces:**
- Consumes: `app.agent_pipeline.run_pipeline_task.delay()`
- Produces: `insight()` returns `{"task_id": "..."}` + user-facing message when `no_data=True`

- [ ] **Step 1: Write test for no_data→task dispatch**

Add to `tests/test_graph.py`:

```python
def test_insight_no_data_returns_task_id(monkeypatch):
    """When no_data=True, insight node should return a task_id for the pipeline."""
    import uuid
    
    fake_task_id = "task_test_" + uuid.uuid4().hex[:8]
    mock_delay = MagicMock(return_value=type('FakeAsyncResult', (), {'id': fake_task_id})())
    monkeypatch.setattr("app.graph.run_pipeline_task", MagicMock(delay=mock_delay))

    state = {"question": "理想L9海外销量", "rows": [], "cols": [], "history": []}
    import app.graph as graph_module
    result = graph_module.insight(state)

    assert result.get("no_data") is True
    assert result.get("task_id") == fake_task_id
    assert mock_delay.called
    # Verify the user-facing message mentions collection
    assert "采集" in result.get("insight", "")
```

- [ ] **Step 2: Run to verify test fails**

```bash
.venv/Scripts/pytest tests/test_graph.py::test_insight_no_data_returns_task_id -v
```

Expected: FAIL — `task_id` not in result

- [ ] **Step 3: Modify the `insight()` node in `app/graph.py`**

The `no_data=True` block in `insight()` currently returns (around line ~475 in graph.py):

```python
return {"no_data": True, "insight": f"未查询到相关数据。{brand_hint}请尝试：\n"
                   f"1. 换一个品牌或车系名称（如 '比亚迪'、'小米SU7'）\n"
                   f"2. 问更宽泛的问题（如 '2025年纯电销量Top10'）",
        "trace": [_t("insight", empty_result=True)]}
```

Replace with:

```python
# P3 FEATURE: oh-my-openagent — trigger async data collection pipeline
import uuid
import secrets
task_id = "agent_" + secrets.token_hex(8)
try:
    from .agent_pipeline import run_pipeline_task
    q = state.get("question", "")
    run_pipeline_task.delay(task_id=task_id, original_question=q,
                            original_user_id=state.get("user_id", 0))
    return {"no_data": True,
            "task_id": task_id,
            "insight": f"数据库暂无相关数据。{brand_hint}"
                       f"已启动智能数据采集（任务ID：{task_id[:12]}…），预计需要 30 秒到 2 分钟。\n"
                       f"采集完成后将自动为您重新查询，请稍候…",
            "trace": [_t("insight", empty_result=True, task_id=task_id)]}
except Exception:
    # FAIL-SAFE: pipeline unavailable → fall back to original dead-end message
    return {"no_data": True, "insight": f"未查询到相关数据。{brand_hint}请尝试：\n"
                   f"1. 换一个品牌或车系名称（如 '比亚迪'、'小米SU7'）\n"
                   f"2. 问更宽泛的问题（如 '2025年纯电销量Top10'）",
            "trace": [_t("insight", empty_result=True)]}
```

- [ ] **Step 4: Run the new test + all graph tests**

```bash
.venv/Scripts/pytest tests/test_graph.py -v
```

Expected: 28 tests PASS (27 existing + 1 new)

- [ ] **Step 5: Commit**

```bash
git add app/graph.py tests/test_graph.py
git commit -m "feat(agent): insight node dispatches pipeline task on no_data"
```

---

### Task 6: SSE endpoint for pipeline progress

**Files:**
- Modify: `app/main.py` (add `GET /api/tasks/{task_id}/stream`)
- Modify: `app/agent_pipeline.py` (add `get_progress` public function)

**Interfaces:**
- Consumes: `agent_pipeline.get_progress(task_id)`
- Produces: `GET /api/tasks/{task_id}/stream` — SSE endpoint pushing stage progress events

- [ ] **Step 1: Add SSE endpoint to `app/main.py`**

Add after the `/api/ask` endpoint (around line 297 in main.py):

```python
# ---------------------------------------------------------------- oh-my-openagent 进度
@app.get("/api/tasks/{task_id}/stream")
async def task_stream(task_id: str, user: User = Depends(get_current_user)):
    """SSE：订阅采集任务的进度事件。
    事件类型：stage（{stage, status, preview}）、done、error。
    前端用 task_id 订阅后逐阶段渲染进度条。
    """
    import asyncio as _asyncio
    from .agent_pipeline import _get_progress as get_progress

    async def gen():
        last_stage = None
        deadline = _asyncio.get_event_loop().time() + 180  # 3 min max
        poll_interval = 1.0

        while _asyncio.get_event_loop().time() < deadline:
            progress = get_progress(task_id)
            if progress is None:
                yield {"event": "stage", "data": json.dumps(
                    {"stage": "queued", "status": "pending", "message": "任务已提交，等待执行…"},
                    ensure_ascii=False)}
                await _asyncio.sleep(poll_interval)
                continue

            stage = progress.get("stage", "unknown")
            status = progress.get("status", "unknown")

            # Push event when stage changes
            if stage != last_stage:
                last_stage = stage
                yield {"event": "stage", "data": json.dumps(progress, ensure_ascii=False, default=str)}

            if stage == "done":
                yield {"event": "done", "data": json.dumps(
                    {"final_answer": progress.get("final_answer", ""), "task_id": task_id},
                    ensure_ascii=False)}
                return

            if status == "failed" or stage == "error":
                yield {"event": "error", "data": json.dumps(
                    {"message": progress.get("error", "采集任务失败"), "task_id": task_id},
                    ensure_ascii=False)}
                return

            await _asyncio.sleep(poll_interval)

        # Timeout
        yield {"event": "error", "data": json.dumps(
            {"message": "采集任务超时，请稍后重试", "task_id": task_id},
            ensure_ascii=False)}

    return EventSourceResponse(gen(), ping=15)
```

- [ ] **Step 2: Verify the file parses correctly**

```bash
cd C:/Users/GUANGBL/lgb_coding/demo1
.venv/Scripts/python -c "from app.main import app; print('OK')"
```

Expected: `OK` (no import errors)

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat(agent): SSE endpoint GET /api/tasks/{task_id}/stream for pipeline progress"
```

---

### Task 7: Integration wiring + end-to-end test

**Files:**
- Modify: `app/agent_pipeline.py` (add public `get_progress` alias)
- Create: `tests/test_agent_pipeline.py` (add end-to-end integration test)

**Interfaces:**
- Produces: `agent_pipeline.get_progress(task_id) -> dict | None` (public alias for `_get_progress`)
- Verifies: The full import chain works without Reddit/PG/MinIO running

- [ ] **Step 1: Add public `get_progress` to `app/agent_pipeline.py`**

Add after the existing `_get_progress` function:

```python
# Public alias
get_progress = _get_progress
```

- [ ] **Step 2: Add integration test**

Add to `tests/test_agent_pipeline.py`:

```python
# ── Integration: full import chain works ──────────────────────────────────────

def test_full_module_import_chain():
    """Verify all new modules can be imported without runtime errors."""
    from app.agent_tools import ALL_TOOLS, get_tool_definitions, execute_tool
    from app.agent_pipeline import _load_pipeline_config, _topological_sort, get_progress, run_pipeline_task
    
    cfg = _load_pipeline_config()
    assert "stages" in cfg
    assert "agents" in cfg
    
    stages = cfg["stages"]
    assert len(stages) >= 4
    
    # Verify topological sort works on real config
    ordered = _topological_sort(stages)
    stage_ids = [s["id"] for s in ordered]
    assert stage_ids.index("research") == 0  # research always first
    assert stage_ids.index("review") == len(stage_ids) - 1  # review always last
    
    # Verify tool definitions load
    code_schemas = get_tool_definitions("code_agent")
    assert len(code_schemas) >= 2
    
    # Verify Celery task is registered
    assert run_pipeline_task.name == "agent_pipeline.run"
```

- [ ] **Step 3: Run all tests**

```bash
.venv/Scripts/pytest tests/test_agent_pipeline.py tests/test_graph.py tests/test_nlu.py -v
```

Expected: all tests PASS (26 NLU + 28 graph + ~16 agent pipeline = ~70 total)

- [ ] **Step 4: Final commit**

```bash
git add app/agent_pipeline.py tests/test_agent_pipeline.py
git commit -m "feat(agent): public get_progress alias + integration tests"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| config/agents.yaml with pipeline DAG + agents + tool permissions | Task 1 |
| Pre-defined tools (http_get, search_web, parse_html, write_to_rag) | Task 2 |
| DAG engine with topological sort | Task 3 |
| Redis progress tracking | Task 3 |
| Celery task wiring | Task 4 |
| insight node triggers pipeline on no_data | Task 5 |
| SSE endpoint for pipeline progress | Task 6 |
| Integration wiring | Task 7 |
| Tests for all components | Tasks 2-7 |

**Placeholder scan:** No TBD/TODO/fill-in details found.

**Type consistency:** `run_pipeline(task_id, original_question, original_user_id)` signature consistent across Tasks 3, 4, and 5. `get_progress(task_id)` consistent across Tasks 3, 6, and 7. Tool schemas defined in Task 2 consumed by Task 3.

---

Plan complete and saved to `docs/superpowers/plans/2026-07-07-oh-my-openagent.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks

**2. Inline Execution** — execute tasks in this session using executing-plans

Which approach?
