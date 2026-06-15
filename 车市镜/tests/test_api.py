"""FastAPI 接口关键路径单测：健康检查可达 + 业务接口强制鉴权。"""
import pytest

pytestmark = pytest.mark.integration   # import 全 app + 起 TestClient，CI 跳过

pytest.importorskip("httpx")          # TestClient 依赖 httpx
from fastapi.testclient import TestClient  # noqa: E402


def _client():
    from app.main import app
    return TestClient(app)


def test_health_ok():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json().get("ok") is True


def test_ask_requires_auth():
    r = _client().post("/api/ask_sync", json={"question": "随便问"})
    assert r.status_code in (401, 403)   # 无 token → 拒绝


def test_history_requires_auth():
    r = _client().get("/api/history")
    assert r.status_code in (401, 403)
