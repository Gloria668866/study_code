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
