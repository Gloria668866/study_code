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

# Default timeout and headers for HTTP requests
_DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
_DEFAULT_HEADERS = {"User-Agent": "CarMirror/1.0 (market research bot; contact@example.com)"}


def _tool_http_get(url: str, **kwargs) -> dict:
    """Fetch a web page. Returns text content and metadata."""
    try:
        resp = httpx.get(url, timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS, follow_redirects=True)
        content = resp.text[:50000]  # truncate to 50KB to avoid overwhelming LLM context
        return {
            "status": "success" if 200 <= resp.status_code < 300 else "failed",
            "status_code": resp.status_code,
            "content": content,
            "content_type": resp.headers.get("content-type", "unknown"),
            "url": str(resp.url),  # final URL after redirects
        }
    except httpx.TimeoutException:
        return {"status": "failed", "error": "Request timeout after 15s", "url": url}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:200], "url": url}


def _tool_search_web(query: str, num_results: int = 5, **kwargs) -> dict:
    """Search the web using DuckDuckGo Instant Answer API (no API key needed).
    For a production system, replace with Bing/Google Search API.
    Returns titles, URLs, and snippets."""
    try:
        # DuckDuckGo Instant Answer API — free, no auth required
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=_DEFAULT_TIMEOUT,
            headers=_DEFAULT_HEADERS,
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
        # RAG_BACKEND=local — write as public document
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
