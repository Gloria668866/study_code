import os
import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcm_kg_app.config import get_settings

settings = get_settings()
API_URL = os.getenv("WEB_API_URL", settings.web_api_url).rstrip("/")

st.set_page_config(page_title="中医药知识图谱增强问答", page_icon="🌿", layout="wide")

st.title("🌿 中医药知识图谱增强问答系统")
st.caption("DeepSeek + FAISS + Neo4j + FastAPI + Streamlit")

with st.sidebar:
    st.header("系统状态")
    try:
        health = requests.get(f"{API_URL}/api/health", timeout=5).json()
        st.success("API 正常")
        st.json(health)
    except Exception as exc:
        st.error(f"API 不可用：{exc}")

    try:
        graph = requests.get(f"{API_URL}/api/graph/status", timeout=5).json()
        if graph.get("available"):
            st.success("Neo4j 可用：GraphRAG 模式")
        else:
            st.warning("Neo4j 不可用：将使用 Local RAG")
        st.json(graph)
    except Exception as exc:
        st.warning(f"无法检查 Neo4j：{exc}")

    prefer_graph = st.checkbox("优先使用 Neo4j GraphRAG", value=True)
    top_k = st.slider("检索资料数量", 1, 10, 5)

examples = [
    "头疼可以了解哪些中医药知识？",
    "蜂蜜有什么功效和禁忌？",
    "小柴胡汤主要用于什么情况？",
    "咳嗽相关的中药有哪些？",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

cols = st.columns(len(examples))
for col, example in zip(cols, examples, strict=False):
    if col.button(example):
        st.session_state.pending_question = example

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("查看检索依据"):
                for source in message["sources"]:
                    st.markdown(f"**{source['name']}** · `{source['type']}` · score={source['score']:.3f}")
                    st.text(source["content"][:800])

question = st.chat_input("请输入你的中医药知识问题...") or st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并调用大模型生成回答..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/chat",
                    json={"question": question, "top_k": top_k, "prefer_graph": prefer_graph},
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
                health = requests.get(f"{API_URL}/api/health", timeout=5).json()
                llm_status = "已配置" if health.get("model_api_configured") else "未配置，使用本地检索归纳"
                st.info(f"当前模式：{data['mode']} | Neo4j 可用：{data['graph_available']} | 大模型：{llm_status}")
                st.markdown(data["answer"])
                st.caption(data["safety_notice"])
                with st.expander("查看检索依据"):
                    for source in data.get("sources", []):
                        st.markdown(f"**{source['name']}** · `{source['type']}` · score={source['score']:.3f}")
                        st.text(source["content"][:800])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                    }
                )
            except Exception as exc:
                st.error(f"请求失败：{exc}")
