# -*- coding:utf-8 -*-
"""Reusable EduRAG question-answering service.

The CLI and Web API both use this module so the demo page shows the same
decision path that the command-line system actually runs.
"""
import os
import sys
import time
import pymysql

from openai import OpenAI

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from base import Config, logger
from mysql_qa import BM25Search, MySQLClient, RedisClient
from rag_qa import RAGSystem, VectorStore


class IntegratedQASystem:
    """Integrated QA system: Redis/BM25 -> intent gate -> RAG/LLM -> fallback."""

    def __init__(self):
        self.logger = logger
        self.config = Config()
        self.mysql_client = MySQLClient()
        self.redis_client = RedisClient()
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

        self.client = OpenAI(
            api_key=self.config.DEEPSEEK_API_KEY,
            base_url=self.config.DEEPSEEK_BASE_URL,
        )
        self.vector_store = VectorStore(
            collection_name=self.config.MILVUS_COLLECTION_NAME,
            host=self.config.MILVUS_HOST,
            port=self.config.MILVUS_PORT,
            database=self.config.MILVUS_DATABASE_NAME,
        )
        self.rag_system = RAGSystem(self.vector_store, self._call_deepseek)
        self._init_conversation_table()

    # ---------- LLM ----------

    def _call_deepseek(self, prompt):
        """Call DeepSeek with streaming output."""
        try:
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt},
                ],
                timeout=60,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"LLM 调用失败: {e}")
            yield f"错误：LLM 调用失败 - {e}"

    # ---------- Conversation history ----------

    def _init_conversation_table(self):
        try:
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            self.mysql_client.connection.commit()
            self.logger.info("对话历史表 (conversations) 初始化成功")
        except pymysql.MySQLError as e:
            self.logger.error(f"初始化对话历史表失败: {e}")
            raise

    def _fetch_recent_history(self, session_id: str) -> list:
        if not session_id:
            return []
        try:
            self.mysql_client.cursor.execute("""
                SELECT question, answer
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id, 5))
            rows = self.mysql_client.cursor.fetchall()
            history = [{"question": row[0], "answer": row[1]} for row in rows]
            return history[::-1]
        except pymysql.MySQLError as e:
            self.logger.error(f"获取对话历史失败: {e}")
            return []

    def get_session_history(self, session_id: str) -> list:
        return self._fetch_recent_history(session_id)

    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        if not session_id:
            return []
        try:
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC LIMIT 5
                    ) AS sub
                )
            """, (session_id, session_id))
            self.mysql_client.connection.commit()
            return self._fetch_recent_history(session_id)
        except pymysql.MySQLError as e:
            self.logger.error(f"更新会话历史失败: {e}")
            self.mysql_client.connection.rollback()
            raise

    # ---------- Event helpers ----------

    def _base_meta(self, query, source_filter, session_id):
        return {
            "route": "start",
            "cache_hit": False,
            "bm25_score": None,
            "bm25_threshold": 0.85,
            "best_question": "",
            "classification": "待判断",
            "need_rag": None,
            "strategy": "待选择",
            "source_filter": source_filter,
            "retrieval_count": 0,
            "docs": [],
            "context_preview": "",
            "latency_ms": 0,
            "session_id": session_id,
            "query": query,
        }

    def _meta_event(self, meta):
        return {"type": "meta", "data": meta, **meta}

    def _done_event(self, meta, start_time):
        meta["latency_ms"] = int((time.time() - start_time) * 1000)
        return {
            "type": "done",
            "done": True,
            "time": round(meta["latency_ms"] / 1000, 2),
            "data": meta,
            **meta,
        }

    # ---------- Query ----------

    def stream_answer(self, query: str, source_filter: str = None, session_id: str = None, history=None):
        """Yield structured events for Web SSE and other observers."""
        start_time = time.time()
        query = (query or "").strip()
        if source_filter and source_filter not in self.config.VALID_SOURCES:
            source_filter = None
        meta = self._base_meta(query, source_filter, session_id)

        if not query:
            meta["route"] = "invalid_query"
            yield {
                "type": "error",
                "error": "empty_query",
                "content": "请输入问题",
                "data": meta,
                **meta,
            }
            return

        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        history = self._fetch_recent_history(session_id) if session_id else (history or [])

        answer, need_rag, bm25_meta = self.bm25_search.search_with_meta(query, threshold=0.85)
        meta.update(bm25_meta)
        meta["need_rag"] = bool(need_rag)
        meta["latency_ms"] = int((time.time() - start_time) * 1000)
        yield self._meta_event(meta.copy())

        if answer:
            meta["classification"] = "标准问答"
            meta["strategy"] = "BM25 精确问答"
            meta["need_rag"] = False
            meta["retrieval_count"] = 0
            collected = answer
            yield self._meta_event(meta.copy())
            yield {"type": "token", "content": answer, "token": answer, **meta}
            if session_id:
                self.update_session_history(session_id, query, collected)
            yield self._done_event(meta, start_time)
            return

        collected = ""
        last_rag_done = None
        try:
            for event in self.rag_system.answer_with_meta(
                query, source_filter=source_filter, history=history
            ):
                event_type = event.get("type")
                if event_type == "meta":
                    rag_meta = event.get("data", {})
                    meta.update(rag_meta)
                    meta["need_rag"] = rag_meta.get("classification") != "通用知识"
                    meta["route"] = "rag" if meta["need_rag"] else "direct_llm"
                    meta["source_filter"] = source_filter
                    meta["latency_ms"] = int((time.time() - start_time) * 1000)
                    yield self._meta_event(meta.copy())
                elif event_type == "token":
                    content = event.get("content", "")
                    collected += content
                    yield {"type": "token", "content": content, "token": content, **meta}
                elif event_type == "done":
                    last_rag_done = event
                elif event_type == "error":
                    yield {"type": "error", "content": event.get("content", "处理失败"), **meta}

            if session_id and collected.strip():
                self.update_session_history(session_id, query, collected)
            if last_rag_done and "time" in last_rag_done:
                meta["rag_time"] = last_rag_done["time"]
            yield self._done_event(meta, start_time)
        except Exception as e:
            self.logger.error(f"统一问答链路失败: {e}")
            fallback = f"抱歉，处理问题时出错，请联系人工客服：{self.config.CUSTOMER_SERVICE_PHONE}"
            meta["route"] = "error"
            yield {"type": "error", "error": str(e), "content": fallback, "data": meta, **meta}
            yield self._done_event(meta, start_time)

    def query(self, query: str, source_filter: str = None, session_id: str = None):
        """Legacy token-only query API used by CLI."""
        for event in self.stream_answer(query, source_filter=source_filter, session_id=session_id):
            if event.get("type") in {"token", "error"}:
                yield event.get("content", "")

    def close(self):
        try:
            self.mysql_client.close()
        except Exception:
            pass
