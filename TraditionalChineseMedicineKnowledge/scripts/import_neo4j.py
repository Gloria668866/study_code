import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcm_kg_app.graph.neo4j_client import Neo4jService


def main() -> None:
    service = Neo4jService()
    result = service.import_graph_records()
    print(result)
    print(service.status())


if __name__ == "__main__":
    main()
