import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcm_kg_app.config import get_settings
from tcm_kg_app.core.data_loader import normalize_extraction_file, save_json


HERB_SOURCE = PROJECT_ROOT / "data" / "extract_herb_data.json"
FORMULA_SOURCE = PROJECT_ROOT / "data" / "extract_formula_data.json"


def main() -> None:
    settings = get_settings()
    all_documents = []
    all_graph_records = []

    for path, dataset in [(HERB_SOURCE, "herb"), (FORMULA_SOURCE, "formula")]:
        if not path.exists():
            raise FileNotFoundError(f"Source data not found: {path}")
        documents, graph_records = normalize_extraction_file(path, dataset)
        all_documents.extend(documents)
        all_graph_records.extend(graph_records)

    dedup = {}
    for document in all_documents:
        dedup[document["id"]] = document

    save_json(settings.documents_path, list(dedup.values()))
    save_json(settings.graph_path, all_graph_records)
    print(f"Prepared {len(dedup)} documents -> {settings.documents_path}")
    print(f"Prepared {len(all_graph_records)} graph records -> {settings.graph_path}")


if __name__ == "__main__":
    main()
