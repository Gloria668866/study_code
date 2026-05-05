"""Data initialization script for EduRAG system.
Imports knowledge documents into Milvus and QA pairs into MySQL, then warms Redis cache.
Usage: python scripts/init_data.py [--milvus-only | --mysql-only | --all]
"""
import sys, os, argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from base import Config, logger
from rag_qa.core.document_processor import process_documents
from rag_qa.core.vector_store import VectorStore
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.retrieval.bm25_search import BM25Search

conf = Config()

DATA_BASE = os.path.join(project_root, "rag_qa", "data")
QA_CSV = os.path.join(DATA_BASE, "all_subjects_qa.csv")
VALID_SOURCES = conf.VALID_SOURCES  # ["ai", "java", "test", "ops", "bigdata"]


def init_milvus():
    """Import all knowledge documents into Milvus."""
    logger.info("=== 开始 Milvus 数据导入 ===")
    vector_store = VectorStore(
        collection_name=conf.MILVUS_COLLECTION_NAME,
        host=conf.MILVUS_HOST,
        port=conf.MILVUS_PORT,
        database=conf.MILVUS_DATABASE_NAME,
    )
    total_chunks = 0
    for source in VALID_SOURCES:
        dir_path = os.path.join(DATA_BASE, f"{source}_data")
        if not os.path.exists(dir_path):
            logger.warning(f"目录不存在，跳过: {dir_path}")
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith(('.md', '.txt', '.pdf', '.docx'))]
        if not files:
            logger.warning(f"目录为空，跳过: {dir_path}")
            continue
        logger.info(f"处理目录 {dir_path}，发现 {len(files)} 个文件")
        try:
            chunks = process_documents(
                dir_path,
                conf.PARENT_CHUNK_SIZE,
                conf.CHILD_CHUNK_SIZE,
                conf.CHUNK_OVERLAP,
            )
            if chunks:
                vector_store.add_documents(chunks)
                total_chunks += len(chunks)
                logger.info(f"  -> 添加 {len(chunks)} 个文档块 (学科: {source})")
            else:
                logger.warning(f"  -> 目录 {dir_path} 未提取到有效文档块")
        except Exception as e:
            logger.error(f"处理目录 {dir_path} 失败: {e}")
    logger.info(f"=== Milvus 数据导入完成，共 {total_chunks} 个文档块 ===")


def init_mysql():
    """Import QA pairs from CSV into MySQL."""
    logger.info("=== 开始 MySQL 数据导入 ===")
    if not os.path.exists(QA_CSV):
        logger.error(f"QA CSV 文件不存在: {QA_CSV}")
        logger.info("请先运行数据生成脚本，或手动准备 CSV 文件")
        return

    mysql = MySQLClient()
    try:
        mysql.create_table()
        mysql.insert_data(QA_CSV)
        logger.info("=== MySQL 数据导入完成 ===")
    except Exception as e:
        logger.error(f"MySQL 数据导入失败: {e}")
    finally:
        mysql.close()


def init_redis():
    """Warm Redis cache from MySQL data."""
    logger.info("=== 开始 Redis 缓存预热 ===")
    try:
        redis_client = RedisClient()
        mysql_client = MySQLClient()
        bm25 = BM25Search(redis_client, mysql_client)
        logger.info("=== Redis 缓存预热完成 ===")
    except Exception as e:
        logger.error(f"Redis 缓存预热失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EduRAG Data Initialization")
    parser.add_argument('--milvus-only', action='store_true')
    parser.add_argument('--mysql-only', action='store_true')
    parser.add_argument('--all', action='store_true', default=True)
    args = parser.parse_args()

    if args.milvus_only:
        init_milvus()
    elif args.mysql_only:
        init_mysql()
        init_redis()
    else:
        init_milvus()
        init_mysql()
        init_redis()
