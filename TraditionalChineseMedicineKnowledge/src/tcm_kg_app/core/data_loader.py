import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TcmDocument:
    id: str
    name: str
    type: str
    content: str
    metadata: dict[str, Any]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _entity_text(entity: dict[str, Any], source_file: str) -> str:
    name = entity.get("name", "")
    entity_type = entity.get("type", "Unknown")
    attributes = entity.get("attributes") or {}
    attr_text = "\n".join(f"{key}: {value}" for key, value in attributes.items() if value)
    return f"名称: {name}\n类型: {entity_type}\n来源文件: {source_file}\n{attr_text}".strip()


def normalize_extraction_file(path: Path, default_dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = load_json(path)
    documents: list[dict[str, Any]] = []
    graph_records: list[dict[str, Any]] = []
    seen_docs: set[tuple[str, str]] = set()

    for item_idx, item in enumerate(raw.get("results", [])):
        filename = item.get("filename", f"{default_dataset}_{item_idx}.txt")
        extract_dict = item.get("extract_dict") or {}
        entities = extract_dict.get("entities") or []
        relationships = extract_dict.get("relationships") or []

        for entity_idx, entity in enumerate(entities):
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "Unknown")).strip() or "Unknown"
            if not name:
                continue
            key = (entity_type, name)
            if key not in seen_docs:
                seen_docs.add(key)
                doc_id = f"{entity_type}:{name}"
                documents.append(
                    {
                        "id": doc_id,
                        "name": name,
                        "type": entity_type,
                        "content": _entity_text(entity, filename),
                        "metadata": {
                            "source_file": filename,
                            "dataset": default_dataset,
                            "attributes": entity.get("attributes") or {},
                        },
                    }
                )

        graph_records.append(
            {
                "id": f"{default_dataset}:{filename}",
                "filename": filename,
                "dataset": default_dataset,
                "entities": entities,
                "relationships": relationships,
            }
        )

    return documents, graph_records


def load_documents(path: Path) -> list[TcmDocument]:
    data = load_json(path)
    return [
        TcmDocument(
            id=item["id"],
            name=item["name"],
            type=item["type"],
            content=item["content"],
            metadata=item.get("metadata") or {},
        )
        for item in data
    ]
