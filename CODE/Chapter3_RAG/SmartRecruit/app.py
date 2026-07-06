# app.py

# --- 0. 导入所需库 ---
# 导入异步编程、文件编码、JSON处理、操作系统交互、系统和时间处理等基础库
import asyncio
import base64
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# 导入 nest_asyncio 库，以解决 Streamlit 环境中 asyncio 事件循环的冲突问题
import nest_asyncio

# 应用补丁，允许事件循环嵌套
nest_asyncio.apply()

# 导入 Streamlit 库，用于构建Web应用界面
import streamlit as st
# 从 LangChain 导入文档和消息结构
from langchain.docstore.document import Document
from langchain_core.messages import AIMessage, HumanMessage
# 导入 loguru 库，用于日志记录
from loguru import logger
# 导入 OpenAI 客户端，用于与 LLM API 交互
from openai import OpenAI

# 从项目其他模块导入自定义的配置、Agent、工具函数和类
from config import config
from rag.rag_pipeline import SmartRecruitAgent
from utils.document_processor import load_and_hash_document, process_document, parse_resume_structure, parser_client
from utils.vector_store import VectorStore

# --- 1. 页面配置与日志过滤 ---
# 设置 Streamlit 页面的基本配置，如浏览器标签页的标题和页面布局
st.set_page_config(page_title="SmartRecruit", layout="wide")


# 定义一个自定义的 stderr 过滤器类，用于屏蔽特定且无害的警告信息
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # 定义需要过滤掉的警告文本
        self.filter_text = "Examining the path of torch.classes raised"

    def write(self, text):
        # 只有当文本不包含过滤词时，才将其写入原始的 stderr
        if self.filter_text not in text:
            self.original_stderr.write(text)
            self.original_stderr.flush()

    def flush(self):
        # 保持 flush 方法以兼容文件对象的接口
        self.original_stderr.flush()


# 应用过滤器：如果当前的 stderr 还不是我们的过滤器实例，就替换它
if not isinstance(sys.stderr, StderrFilter):
    sys.stderr = StderrFilter(sys.stderr)


# --- 2. 核心组件的缓存与会话状态管理 ---
# 使用 Streamlit 的缓存装饰器 @st.cache_resource
# 这可以确保函数只在第一次被调用时执行一次，后续调用直接返回缓存的结果。
# 这对于加载昂贵的资源（如模型、数据库连接）至关重要，可以极大提升应用性能。
@st.cache_resource
def get_vector_store():
    """初始化并返回 VectorStore 实例。此函数会被缓存。"""
    logger.info("首次初始化 VectorStore...")
    return VectorStore()


@st.cache_resource
def get_openai_client():
    """初始化并返回用于文档处理的 OpenAI 客户端。此函数会被缓存。"""
    logger.info("首次初始化 OpenAI Client for Doc Processing...")
    return OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def initialize_session_state():
    """
    初始化 Streamlit 的会话状态 (session_state)。
    session_state 是一个类似字典的对象，用于在用户的多次交互（页面刷新）之间保持数据。
    """
    # 如果会话中还没有 agent 实例，则创建一个新的
    if "agent" not in st.session_state:
        st.session_state.agent = SmartRecruitAgent()
    # 如果会话中还没有消息列表，则初始化为空列表
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 如果会话中还没有上传日志列表，则初始化为空列表
    if "upload_logs" not in st.session_state:
        st.session_state.upload_logs = []
    # [关键] 用于存储上一轮AI推荐的候选人列表，以实现多轮对话中的上下文记忆
    if "last_candidates" not in st.session_state:
        st.session_state.last_candidates = None


# 应用启动时，立即执行会话状态的初始化
initialize_session_state()

# 尝试加载所有核心组件，如果失败则显示错误信息并停止应用
try:
    vector_store = get_vector_store()
    doc_client = get_openai_client()
    agent = st.session_state.agent
except Exception as e:
    st.error(f"加载核心组件失败: {e}")
    logger.critical(f"核心组件加载失败: {e}", exc_info=True)
    st.stop()


# --- 3. 简历查看与渲染逻辑 ---
def render_resume_view(doc_hash: str):
    """
    根据给定的简历哈希值，渲染单个简历的详细视图。
    这个视图通常是通过点击主界面上的“查看简历”链接触发的。
    """
    try:
        # 1. 从数据库获取简历的元数据
        metadata = vector_store.get_metadata_by_hash(doc_hash)
        if not metadata:
            st.error("无法找到该简历的元数据。")
            if st.button("返回"):
                st.query_params.clear()  # 清除URL参数
                st.rerun()  # 重新运行脚本以返回主页
            return

        # 2. 提取文件路径并显示标题
        file_path = metadata.get("metadata", {}).get("file_path", "")
        st.header(f"查看简历: {os.path.basename(file_path)}")
        st.caption(f"Hash: {doc_hash}")

        # 3. 根据文件类型渲染简历内容
        if os.path.exists(file_path) and file_path.endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            # 如果是PDF文件，则以Base64编码嵌入到iframe中进行展示
            if file_path.endswith('.pdf'):
                with open(file_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            # 如果是图片文件，则直接显示图片
            else:
                st.image(file_path)
        else:
            # 如果文件不是支持的格式或路径不存在，则显示从数据库中提取的纯文本内容
            st.subheader("简历文本内容")
            content = vector_store.get_full_resume(doc_hash)
            st.markdown(content or "无法加载简历文本内容。")

        # 4. 提供返回按钮
        if st.button("返回推荐列表"):
            st.query_params.clear()
            st.rerun()

    except Exception as e:
        st.error(f"渲染简历时出错: {e}")
        logger.error(f"渲染简历失败, hash: {doc_hash}", exc_info=True)
        if st.button("返回"): st.query_params.clear(); st.rerun()


# --- 4. 主应用界面与交互逻辑 ---
def run_async(coro):
    """一个辅助函数，用于在同步的Streamlit环境中运行异步代码。"""
    try:
        # 尝试获取当前正在运行的事件循环
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 如果没有正在运行的事件循环，则创建一个新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    # 运行异步协程直到完成
    return loop.run_until_complete(coro)


def render_candidate_cards(candidates: List[Dict[str, Any]], title: str = "根据您的需求，我推荐以下候选人："):
    """一个UI函数，用于将候选人列表渲染成信息卡片。"""
    if title:
        st.markdown(title)

    # 如果候选人列表为空，显示提示信息
    if not candidates:
        st.info("筛选后没有找到完全匹配的候选人。")
        return

    # 遍历每个候选人，为他们创建一个带边框的容器
    for i, candidate in enumerate(candidates, 1):
        with st.container(border=True):
            doc_hash = candidate.get('doc_hash', '')
            file_path = candidate.get('file_path', 'N/A')

            # 显示候选人编号和文件名
            st.subheader(f"候选人 {i}: {os.path.basename(file_path)}")
            # 显示AI给出的推荐理由
            st.markdown(f"**推荐理由:** {candidate.get('reason', 'N/A')}")

            # 如果有哈希值，则生成一个可以点击的链接，通过URL参数导航到简历视图
            if doc_hash:
                st.markdown(f'<a href="?view_resume={doc_hash}" target="_blank">点击在线查看简历</a>',
                            unsafe_allow_html=True)
            else:
                st.warning("无法生成简历链接 (缺少 doc_hash).")


def render_main_chat_ui():
    """渲染主聊天界面，包含推荐和上传两个功能区。"""
    st.title("SmartRecruit 智能简历推荐系统")

    # 创建两个标签页
    tab1, tab2 = st.tabs(["候选人推荐", "简历上传"])

    # --- 候选人推荐标签页 ---
    with tab1:
        st.header("智能招聘助手")
        # 遍历并渲染整个聊天历史
        for message in st.session_state.messages:
            # 根据消息类型（'user'或'assistant'）创建不同的聊天气泡
            with st.chat_message(message.type):
                # 如果是AI的消息，需要特殊处理，因为它可能是结构化的JSON
                if isinstance(message, AIMessage):
                    content_data = None
                    try:
                        # 尝试将消息内容解析为JSON
                        content_data = json.loads(message.content)
                    except (json.JSONDecodeError, TypeError):
                        # 如果解析失败，则当作纯文本处理
                        st.markdown(message.content)
                        continue

                    # 如果解析成功且是字典格式
                    if isinstance(content_data, dict):
                        response_text = content_data.get("response")
                        candidates = content_data.get("candidates")

                        # 首先显示AI的自然语言回答
                        if isinstance(response_text, str):
                            st.markdown(response_text)

                        # 如果返回了候选人列表，则调用函数渲染卡片
                        if isinstance(candidates, list):
                            render_candidate_cards(candidates, title=None)  # title=None避免重复打印标题

                        # 如果既没有文本也没有候选人，显示默认消息
                        elif not response_text and not candidates:
                            st.markdown("抱歉，我暂时无法回答这个问题。")

                    else:  # 兼容内容是纯文本的AIMessage
                        st.markdown(message.content)
                else:  # 如果是用户的消息，直接显示
                    st.markdown(message.content)

        # 在页面底部创建聊天输入框，等待用户输入
        if prompt := st.chat_input("请输入您的招聘需求..."):
            # 1. 将用户输入添加到消息历史中
            st.session_state.messages.append(HumanMessage(content=prompt))
            # 2. 在界面上显示用户的输入
            with st.chat_message("user"):
                st.markdown(prompt)

            # 3. 创建一个AI的聊天气泡，并显示加载动画
            with st.chat_message("assistant"):
                with st.spinner("AI 正在分析您的需求并匹配简历..."):
                    try:
                        # 4. [核心] 调用Agent的arun方法，传入用户输入和上一轮的候选人列表（上下文）
                        agent_response = run_async(agent.arun(prompt, st.session_state.last_candidates))

                        # 5. 将Agent返回的完整结构化响应（JSON字符串）添加到消息历史
                        response_content_str = json.dumps(agent_response, ensure_ascii=False)
                        st.session_state.messages.append(AIMessage(content=response_content_str))

                        # 6. [关键] 更新上下文：如果Agent返回了候选人列表（即使是空列表），
                        #    就用它更新session_state中的last_candidates，为下一次追问做准备
                        if agent_response.get("candidates") is not None:
                            st.session_state.last_candidates = agent_response["candidates"]

                        # 7. 重新运行整个脚本，以刷新界面并显示AI的最新回复
                        st.rerun()
                    except Exception as e:
                        st.error(f"处理您的请求时出错: {e}")
                        logger.error(f"Agent run failed: {e}", exc_info=True)

    # --- 简历上传标签页 ---
    with tab2:
        st.header("简历上传")
        # 创建文件上传组件
        uploaded_file = st.file_uploader("上传简历", type=["md", "docx", "pdf", "txt", "jpg", "png"])

        # 如果用户上传了文件
        if uploaded_file:
            # 确保本地简历存储目录存在
            resume_dir = config.LOCAL_RESUME_DIR
            os.makedirs(resume_dir, exist_ok=True)
            # 创建一个安全且唯一的文件名，避免重名和路径问题
            safe_filename = f"{os.path.splitext(uploaded_file.name)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}{os.path.splitext(uploaded_file.name)[1]}"
            file_path = os.path.join(resume_dir, safe_filename)

            # 将上传的文件内容写入到本地磁盘
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            try:
                # 显示处理中的加载动画
                with st.spinner(f"正在处理简历: {uploaded_file.name}..."):
                    # [数据入库流程]
                    # 1. 加载文件内容并计算哈希值
                    doc_content, doc_hash = load_and_hash_document(file_path, doc_client)
                    # 2. 从文本中提取结构化信息
                    structured_data = parse_resume_structure(doc_content, parser_client)
                    # 3. 创建LangChain的Document对象
                    doc = Document(
                        page_content=doc_content,
                        metadata={
                            "file_path": file_path, "original_name": uploaded_file.name,
                            "hash": doc_hash, "timestamp": datetime.now().isoformat(),
                            **structured_data
                        }
                    )
                    # 4. 将文档切分成小块
                    chunks = process_document(doc)
                    # 5. 将所有处理好的数据存入数据库（Milvus, ES, MongoDB）
                    success = vector_store.store_resume(doc, chunks, structured_data)

                # 根据存储结果向用户显示反馈
                if success:
                    st.success(f"简历上传并处理成功: {uploaded_file.name}")
                    st.session_state.upload_logs.append(f"成功: {uploaded_file.name} (hash: {doc_hash})")
                else:
                    st.warning(f"简历已存在: {uploaded_file.name}")
                    st.session_state.upload_logs.append(f"重复: {uploaded_file.name} (hash: {doc_hash})")
                    os.remove(file_path)  # 如果是重复文件，删除本地保存的副本
            except Exception as e:
                # 如果处理过程中发生任何错误，向用户显示错误信息
                st.error(f"处理失败: {str(e)}")
                st.session_state.upload_logs.append(f"失败: {uploaded_file.name}, 错误: {str(e)}")
                logger.error(f"简历处理失败: {uploaded_file.name}", exc_info=True)
                if os.path.exists(file_path):
                    os.remove(file_path)  # 处理失败也删除文件

        # 显示上传日志区域
        st.subheader("上传日志")
        if st.session_state.get("upload_logs"):
            st.text_area("Logs", "\n".join(st.session_state.upload_logs), height=200)


# --- 5. 主程序入口 ---
# 这是一个简单的路由逻辑，根据URL中的查询参数来决定显示哪个页面
if "view_resume" in st.query_params:
    # 如果URL是 .../?view_resume=xxx，则渲染单个简历视图
    render_resume_view(st.query_params["view_resume"])
else:
    # 否则，渲染主聊天界面
    render_main_chat_ui()