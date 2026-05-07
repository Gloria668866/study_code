from typing import Any

from neo4j import GraphDatabase

from tcm_kg_app.config import get_settings
from tcm_kg_app.core.data_loader import load_json
from tcm_kg_app.schemas.chat import GraphStatusResponse


class Neo4jService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.driver = None

    def connect(self) -> None:
        if self.driver is None:
            if not self.settings.neo4j_password:
                raise RuntimeError("NEO4J_PASSWORD is empty")
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def status(self) -> GraphStatusResponse:
        try:
            self.connect()
            assert self.driver is not None
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            return GraphStatusResponse(
                available=True,
                message="Neo4j is available",
                node_count=node_count,
                relationship_count=rel_count,
            )
        except Exception as exc:
            return GraphStatusResponse(available=False, message=str(exc))

    def import_graph_records(self) -> dict[str, int]:
        self.connect()
        assert self.driver is not None
        records = load_json(self.settings.graph_path)
        node_count = 0
        rel_count = 0
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT tcm_entity_key IF NOT EXISTS FOR (n:TcmEntity) REQUIRE (n.type, n.name) IS UNIQUE")
            for record in records:
                entities = record.get("entities") or []
                relationships = record.get("relationships") or []
                for entity in entities:
                    name = str(entity.get("name", "")).strip()
                    entity_type = str(entity.get("type", "Unknown")).strip() or "Unknown"
                    if not name:
                        continue
                    attributes = entity.get("attributes") or {}
                    session.run(
                        """
                        MERGE (n:TcmEntity {type: $type, name: $name})
                        SET n += $attributes,
                            n.source_file = $source_file,
                            n.dataset = $dataset
                        """,
                        type=entity_type,
                        name=name,
                        attributes=attributes,
                        source_file=record.get("filename"),
                        dataset=record.get("dataset"),
                    )
                    node_count += 1
                for relationship in relationships:
                    rel_count += self._merge_relationship(session, relationship)
        return {"nodes_processed": node_count, "relationships_processed": rel_count}

    @staticmethod
    def _merge_relationship(session: Any, relationship: dict[str, Any]) -> int:
        source = relationship.get("source") or relationship.get("start") or relationship.get("from")
        target = relationship.get("target") or relationship.get("end") or relationship.get("to")
        rel_type = relationship.get("type") or relationship.get("relationship") or "RELATED_TO"
        if not source or not target:
            return 0
        source_name = source.get("name") if isinstance(source, dict) else str(source)
        source_type = source.get("type", "Unknown") if isinstance(source, dict) else "Unknown"
        target_name = target.get("name") if isinstance(target, dict) else str(target)
        target_type = target.get("type", "Unknown") if isinstance(target, dict) else "Unknown"
        safe_rel_type = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(rel_type).upper()) or "RELATED_TO"
        session.run(
            f"""
            MERGE (a:TcmEntity {{type: $source_type, name: $source_name}})
            MERGE (b:TcmEntity {{type: $target_type, name: $target_name}})
            MERGE (a)-[r:{safe_rel_type}]->(b)
            SET r.source = $source
            """,
            source_type=source_type,
            source_name=source_name,
            target_type=target_type,
            target_name=target_name,
            source="extraction_json",
        )
        return 1

    def find_related(self, names: list[str], limit: int = 20) -> list[dict[str, Any]]:
        self.connect()
        assert self.driver is not None
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:TcmEntity)-[r]-(m:TcmEntity)
                WHERE n.name IN $names
                RETURN n.name AS source, n.type AS source_type,
                       type(r) AS relation,
                       m.name AS target, m.type AS target_type
                LIMIT $limit
                """,
                names=names,
                limit=limit,
            )
            return [record.data() for record in result]
