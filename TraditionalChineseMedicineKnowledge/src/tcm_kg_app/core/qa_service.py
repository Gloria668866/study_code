from tcm_kg_app.graph.neo4j_client import Neo4jService
from tcm_kg_app.llm.deepseek_client import DeepSeekClient, build_medical_safety_notice
from tcm_kg_app.rag.vector_store import VectorStore
from tcm_kg_app.schemas.chat import ChatResponse, RetrievedDocument


class QaService:
    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.llm = DeepSeekClient()
        self.graph = Neo4jService()

    def answer(self, question: str, top_k: int = 5, prefer_graph: bool = True) -> ChatResponse:
        sources = self._safe_search(question, top_k)
        graph_available = False
        graph_context = ""
        mode = "local_rag"

        if prefer_graph:
            status = self.graph.status()
            graph_available = status.available
            if status.available and sources:
                try:
                    related = self.graph.find_related([doc.name for doc in sources[:3]], limit=20)
                    if related:
                        mode = "graph_rag"
                        graph_context = "\n".join(
                            f"{item['source']}({item['source_type']}) -[{item['relation']}]- {item['target']}({item['target_type']})"
                            for item in related
                        )
                except Exception:
                    graph_available = False

        answer = self._generate_answer(question, sources, graph_context)
        return ChatResponse(
            answer=answer,
            mode=mode,  # type: ignore[arg-type]
            sources=sources,
            graph_available=graph_available,
            safety_notice=build_medical_safety_notice(),
        )

    def _safe_search(self, question: str, top_k: int) -> list[RetrievedDocument]:
        try:
            return self.vector_store.search(question, top_k=top_k)
        except Exception:
            return []

    def _generate_answer(self, question: str, sources: list[RetrievedDocument], graph_context: str) -> str:
        context = "\n\n".join(
            f"[资料{i + 1}] 名称：{doc.name}\n类型：{doc.type}\n内容：{doc.content}"
            for i, doc in enumerate(sources)
        )
        if graph_context:
            context += f"\n\n[知识图谱关系]\n{graph_context}"

        if not context:
            context = "当前没有检索到本地中医药资料，请基于通用知识谨慎回答，并说明信息不足。"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个中医药知识图谱问答助手。请只提供知识解释、资料归纳和就医提醒，"
                    "不要给出确定性诊断、处方剂量或替代医生的治疗建议。"
                    "回答要结构清晰，优先引用给定资料。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"可参考资料：\n{context}\n\n"
                    "请用中文回答，包含：1. 简要结论；2. 相关中医药知识；3. 风险与就医提醒。"
                ),
            },
        ]
        answer = self.llm.chat(messages)
        if answer:
            return answer
        return self._build_local_answer(question, sources, graph_context)

    @staticmethod
    def _build_local_answer(question: str, sources: list[RetrievedDocument], graph_context: str) -> str:
        if not sources:
            return (
                f"## 简要结论\n当前本地索引中没有检索到与“{question}”直接相关的资料。\n\n"
                "## 相关中医药知识\n请先确认数据集中是否包含该方剂/中药/症状，并重新运行 `scripts/prepare_data.py` 和 `scripts/build_index.py` 更新索引。\n\n"
                "## 风险与就医提醒\n本地资料不足时，不建议据此自行判断病情或用药；如有持续不适，请咨询专业医生。"
            )

        lines = [
            "## 简要结论",
            "当前未配置大模型 API Key，系统已改用本地索引检索结果进行资料归纳。以下内容来自本地知识库，供知识查询参考。",
            "",
            "## 相关中医药知识",
        ]
        for i, doc in enumerate(sources, start=1):
            snippet = doc.content.strip().replace("\r\n", "\n")
            if len(snippet) > 500:
                snippet = f"{snippet[:500]}..."
            lines.extend(
                [
                    f"{i}. **{doc.name}**（{doc.type}，相似度 {doc.score:.3f}）",
                    snippet,
                    "",
                ]
            )

        if graph_context:
            lines.extend(["## 知识图谱关系", graph_context, ""])

        lines.extend(
            [
                "## 风险与就医提醒",
                "以上是本地资料归纳，不构成诊断、处方或治疗建议。涉及具体病情、剂量、儿童/孕产妇/老人/慢病患者用药时，请咨询专业医生。",
            ]
        )
        return "\n".join(lines)
