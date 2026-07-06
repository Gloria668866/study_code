import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcm_kg_app.rag.vector_store import VectorStore


def main() -> None:
    count = VectorStore().build()
    print(f"Built FAISS index for {count} documents.")


if __name__ == "__main__":
    main()
