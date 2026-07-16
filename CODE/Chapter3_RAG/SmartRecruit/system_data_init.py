
# system_data_init.py
import os
import time
from loguru import logger
from config import config
from utils.document_processor import load_and_hash_document, parse_resume_structure, process_document, parser_client
from utils.vector_store import VectorStore
from openai import OpenAI
from langchain.docstore.document import Document
import tqdm
from datetime import datetime

# --- 日志配置 ---
# 将初始化脚本的日志单独存放，方便追踪
init_log_path = os.path.join(config.LOG_DIR, "system_init.log")
logger.add(init_log_path, rotation="10 MB", encoding="utf-8", level="INFO")

def initialize_system_data():
    """
    系统初始化主函数：
    1. 扫描本地简历目录。
    2. 提取文本内容和结构化信息。
    3. 切块并存入所有数据库 (Milvus, ES, MongoDB)。
    """
    logger.info("="*50)
    logger.info("开始执行系统数据初始化...")
    logger.info(f"扫描简历目录: {config.LOCAL_RESUME_DIR}")

    if not os.path.exists(config.LOCAL_RESUME_DIR):
        logger.error(f"错误：简历目录 '{config.LOCAL_RESUME_DIR}' 不存在。")
        return

    # 初始化核心组件
    try:
        vector_store = VectorStore()
        # 用于文档加载（可能含图片）的客户端
        doc_client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    except Exception as e:
        logger.critical(f"初始化核心组件失败: {e}", exc_info=True)
        return

    supported_extensions = [".md", ".docx", ".pdf", ".txt", ".jpg", ".png"]
    resume_files = [f for f in os.listdir(config.LOCAL_RESUME_DIR) if os.path.splitext(f)[1].lower() in supported_extensions]
    
    if not resume_files:
        logger.warning("未在目录中找到任何支持的简历文件。")
        return

    logger.info(f"发现 {len(resume_files)} 个简历文件待处理。")
    
    success_count = 0
    fail_count = 0
    exist_count = 0

    # 使用tqdm创建进度条
    with tqdm.tqdm(total=len(resume_files), desc="处理简历") as pbar:
        for filename in resume_files:
            file_path = os.path.join(config.LOCAL_RESUME_DIR, filename)
            pbar.set_description(f"处理中: {filename}")
            try:
                # 1. 加载和哈希
                doc_content, doc_hash = load_and_hash_document(file_path, doc_client)
                
                # 2. 解析结构化数据
                structured_data = parse_resume_structure(doc_content, parser_client)
                
                # 3. 创建文档对象
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        "file_path": file_path, "original_name": filename,
                        "hash": doc_hash, "timestamp": datetime.now().isoformat(),
                        **structured_data
                    }
                )
                
                # 4. 切块
                chunks = process_document(doc)
                
                # 5. 存储
                success = vector_store.store_resume(doc, chunks, structured_data)
                
                if success:
                    logger.info(f"成功处理并存储: {filename}")
                    success_count += 1
                else:
                    logger.warning(f"简历已存在，跳过: {filename}")
                    exist_count += 1
                
            except Exception as e:
                logger.error(f"处理文件失败: {filename}, 错误: {e}", exc_info=True)
                fail_count += 1
            
            pbar.update(1)
            time.sleep(0.1) # 防止日志和进度条刷新过快

    # --- 总结报告 ---
    logger.info("="*50)
    logger.info("系统数据初始化完成！")
    logger.info(f"处理结果总结:")
    logger.info(f"  - 成功新增: {success_count} 份")
    logger.info(f"  - 已存在 (跳过): {exist_count} 份")
    logger.info(f"  - 处理失败: {fail_count} 份")
    logger.info(f"详细日志请查看: {init_log_path}")
    logger.info("="*50)


if __name__ == "__main__":
    """
    手动执行系统初始化脚本。
    在执行前，请确保：
    1. Milvus, Elasticsearch, MongoDB 服务已启动。
    2. info.py 中的配置正确。
    3. 简历文件已放置在 config.LOCAL_RESUME_DIR 目录中。
    """
    print("即将开始系统数据初始化...")
    print("这将扫描、解析并存储所有本地简历。")
    initialize_system_data()
    print("初始化流程结束.")

