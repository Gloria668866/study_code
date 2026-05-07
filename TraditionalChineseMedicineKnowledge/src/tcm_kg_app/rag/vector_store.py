import json
import shutil
import tempfile
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from tcm_kg_app.config import get_settings
from tcm_kg_app.core.data_loader import TcmDocument, load_documents
from tcm_kg_app.schemas.chat import RetrievedDocument


class VectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.documents: list[TcmDocument] = []

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.settings.embedding_model_path)
        return self.model

    def build(self, documents_path: Path | None = None) -> int:
        documents_path = documents_path or self.settings.documents_path
        print(f"Loading documents from {documents_path}...", flush=True)
        documents = load_documents(documents_path)
        print(f"Loaded {len(documents)} documents.", flush=True)

        print(f"Loading embedding model: {self.settings.embedding_model_path}", flush=True)
        model = self._get_model()
        texts = [doc.content for doc in documents]

        print("Encoding documents and building FAISS index...", flush=True)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=16,
        )
        vectors = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        index_parent = self.settings.faiss_index_path.parent
        meta_parent = self.settings.faiss_meta_path.parent
        index_parent.mkdir(parents=True, exist_ok=True)
        meta_parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving FAISS index to {self.settings.faiss_index_path}...", flush=True)
        print(f"Saving metadata to {self.settings.faiss_meta_path}...", flush=True)

        try:
            with tempfile.TemporaryDirectory(prefix="tcm_faiss_") as temp_dir:
                temp_index_path = Path(temp_dir) / "tcm.faiss"
                faiss.write_index(index, str(temp_index_path))
                shutil.copyfile(temp_index_path, self.settings.faiss_index_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save FAISS index to {self.settings.faiss_index_path}. "
                f"Parent exists: {index_parent.exists()}. "
                "On Windows, FAISS may fail with non-ASCII paths, so the code writes "
                "through an ASCII temporary path first."
            ) from exc

        with self.settings.faiss_meta_path.open("w", encoding="utf-8") as file:
            json.dump([doc.__dict__ for doc in documents], file, ensure_ascii=False, indent=2)
        print(f"Saved FAISS index to {self.settings.faiss_index_path}", flush=True)
        return len(documents)

    def load(self) -> None:
        if self.index is not None and self.documents:
            return
        if not self.settings.faiss_index_path.exists() or not self.settings.faiss_meta_path.exists():
            raise FileNotFoundError("FAISS index not found. Please run scripts/build_index.py first.")
        self.index = faiss.read_index(str(self.settings.faiss_index_path))
        with self.settings.faiss_meta_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        self.documents = [TcmDocument(**item) for item in data]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        self.load()
        if self.index is None:
            return []
        model = self._get_model()
        embedding = model.encode([query], normalize_embeddings=True)
        query_vector = np.asarray(embedding, dtype="float32")
        scores, indices = self.index.search(query_vector, top_k)

        results: list[RetrievedDocument] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                RetrievedDocument(
                    id=doc.id,
                    name=doc.name,
                    type=doc.type,
                    score=float(score),
                    content=doc.content,
                    metadata=doc.metadata,
                )
            )
        return results
