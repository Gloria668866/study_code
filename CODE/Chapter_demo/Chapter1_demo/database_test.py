import pymysql
from pymongo import MongoClient
from pymilvus import connections, Collection, utility
from elasticsearch import Elasticsearch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mysql_connection(host, user, password, database, port=3306):
    """测试 MySQL 数据库连接"""
    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        # 测试连接
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION() as version")
            result = cursor.fetchone()
            logger.info(f"MySQL连接成功! 版本: {result['version']}")

        connection.close()
        return True

    except Exception as e:
        logger.error(f"MySQL连接失败: {str(e)}")
        return False


def test_mongodb_connection(host, port=27017, username=None, password=None, database="admin"):
    """测试 MongoDB 数据库连接"""
    try:
        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}/{database}"
        else:
            uri = f"mongodb://{host}:{port}"

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # 测试连接
        server_info = client.server_info()
        logger.info(f"MongoDB连接成功! 版本: {server_info['version']}")

        client.close()
        return True

    except Exception as e:
        logger.error(f"MongoDB连接失败: {str(e)}")
        return False


def test_milvus_connection(host, port=19530):
    """测试 Milvus 向量数据库连接"""
    try:
        connections.connect(host=host, port=port)

        # 获取版本信息测试连接
        version = utility.get_server_version()
        logger.info(f"Milvus连接成功! 版本: {version}")

        connections.disconnect()
        return True

    except Exception as e:
        logger.error(f"Milvus连接失败: {str(e)}")
        return False


def test_elasticsearch_connection(host, port=9200, username=None, password=None):
    """测试 Elasticsearch 连接"""
    try:
        if username and password:
            es = Elasticsearch(
                [f"{host}:{port}"],
                http_auth=(username, password)
            )
        else:
            es = Elasticsearch([f"{host}:{port}"])

        # 测试连接
        info = es.info()
        logger.info(f"Elasticsearch连接成功! 集群名称: {info['cluster_name']}, 版本: {info['version']['number']}")

        return True

    except Exception as e:
        logger.error(f"Elasticsearch连接失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 配置数据库连接参数 - 请根据实际情况修改
    mysql_config = {
        "host": "localhost",
        "user": "root",
        "password": "your_mysql_password",
        "database": "test_db",
        "port": 3306
    }

    mongodb_config = {
        "host": "localhost",
        "port": 27017,
        "username": None,  # 如果有认证，请填写用户名
        "password": None,  # 如果有认证，请填写密码
        "database": "admin"
    }

    milvus_config = {
        "host": "localhost",
        "port": 19530
    }

    es_config = {
        "host": "localhost",
        "port": 9200,
        "username": None,  # 如果有认证，请填写用户名
        "password": None  # 如果有认证，请填写密码
    }

    # 测试所有连接
    logger.info("开始测试数据库连接...")

    mysql_success = test_mysql_connection(**mysql_config)
    mongodb_success = test_mongodb_connection(**mongodb_config)
    milvus_success = test_milvus_connection(**milvus_config)
    es_success = test_elasticsearch_connection(**es_config)

    # 输出总结报告
    logger.info("=" * 50)
    logger.info("连接测试总结:")
    logger.info(f"MySQL: {'成功' if mysql_success else '失败'}")
    logger.info(f"MongoDB: {'成功' if mongodb_success else '失败'}")
    logger.info(f"Milvus: {'成功' if milvus_success else '失败'}")
    logger.info(f"Elasticsearch: {'成功' if es_success else '失败'}")