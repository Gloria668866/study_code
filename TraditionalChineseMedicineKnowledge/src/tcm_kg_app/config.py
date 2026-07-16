import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    model_api_key: str = _env("MODEL_API_KEY")
    model_base_url: str = _env("MODEL_BASE_URL", "https://api.deepseek.com/v1")
    model_name: str = _env("MODEL_NAME", "deepseek-v4-pro")

    embedding_model_path: str = _env(
        "EMBEDDING_MODEL_PATH",
        str(PROJECT_ROOT / "model" / "bge-large-zh-v1.5"),
    )

    neo4j_uri: str = _env("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = _env("NEO4J_USER", "neo4j")
    neo4j_password: str = _env("NEO4J_PASSWORD")

    api_host: str = _env("API_HOST", "0.0.0.0")
    api_port: int = _env_int("API_PORT", 8000)
    web_api_url: str = _env("WEB_API_URL", "http://localhost:8000")
    top_k: int = _env_int("TOP_K", 5)

    data_dir: Path = PROJECT_ROOT / "data"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "indexes"

    documents_path: Path = PROJECT_ROOT / "data" / "processed" / "documents.json"
    graph_path: Path = PROJECT_ROOT / "data" / "processed" / "graph_records.json"
    faiss_index_path: Path = PROJECT_ROOT / "data" / "indexes" / "tcm.faiss"
    faiss_meta_path: Path = PROJECT_ROOT / "data" / "indexes" / "tcm_meta.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
