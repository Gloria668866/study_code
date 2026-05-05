# info.py.py
import os
from pydantic import BaseModel, Field


class Config(BaseModel):
    """系统配置类，使用Pydantic确保类型安全和验证"""
    # MySQL 配置
    MYSQL_HOST: str = Field(default="localhost", description="MySQL 主机")
    MYSQL_USER: str = Field(default="root", description="MySQL 用户")
    MYSQL_PASSWORD: str = Field(default="YOUR_MYSQL_PASSWORD", description="MySQL 密码")
    MYSQL_DB: str = Field(default="resume_rag_db", description="MySQL 数据库")

    # MongoDB 配置
    MONGO_HOST: str = Field(default="82.156.249.211", description="MongoDB 主机")
    MONGO_PORT: int = Field(default=27017, description="MongoDB 端口")
    MONGO_DB: str = Field(default="resume_db", description="MongoDB 数据库")
    MONGO_USER: str = Field(default="admin", description="MongoDB 用户")
    MONGO_PASSWORD: str = Field(default="YOUR_MYSQL_PASSWORD", description="MongoDB 密码")

    # Milvus 配置
    MILVUS_HOST: str = Field(default="82.156.249.211", description="Milvus 主机")
    MILVUS_PORT: str = Field(default="19530", description="Milvus 端口")
    MILVUS_COLLECTION_NAME: str = Field(default="resume_embeddings", description="Milvus 集合名")
    MILVUS_DIMENSION: int = Field(default=1024, description="向量维度，bge-m3")  # 更新为bge-m3维度

    # Elasticsearch 配置
    ES_HOST: str = Field(default="http://82.156.249.211:9200", description="ES 主机")
    ES_INDEX_NAME: str = Field(default="resume_chunks", description="ES 索引名")

    # 模型路径
    MODEL_PATH: str = Field(default="D:\\LLM_Codes\\Chapter3_RAG\\app_new\\models", description="本地模型路径")
    EMBEDDING_MODEL: str = Field(default=r"D:\LLM_Codes\Chapter3_RAG\SmartRecruit\models\bge-m3",
                                 description="嵌入模型")
    RERANKER_MODEL: str = Field(default=r"D:\LLM_Codes\Chapter3_RAG\SmartRecruit\models\bge-m3",
                                description="重排序模型")

    # API 密钥
    # DASHSCOPE_API_KEY: str = Field(default="YOUR_DASHSCOPE_API_KEY_HERE", description="DashScope API 密钥")
    DASHSCOPE_API_KEY: str = Field(default="YOUR_DASHSCOPE_API_KEY_HERE", description="DashScope API 密钥")

    # 本地简历路径
    LOCAL_RESUME_DIR: str = Field(default="D:\\LLM_Codes\\Chapter3_RAG\\SmartRecruit\\data\\resume",
                                  description="本地简历目录")

    # 日志路径
    LOG_DIR: str = Field(default="logs", description="日志目录")

    # 临时文件路径（用于上传文件）
    TEMP_DIR: str = Field(default="temp", description="临时文件目录")

    # 切分参数
    PARENT_CHUNK_SIZE: int = Field(default=1000, description="父块大小")
    CHILD_CHUNK_SIZE: int = Field(default=400, description="子块大小")
    CHUNK_OVERLAP: int = Field(default=100, description="块重叠大小")
    RETRIEVAL_K: int = Field(default=20, description="检索返回数量")  # 增加k以支持BM25合并
    CANDIDATE_M: int = Field(default=2, description="默认推荐候选人数量")


config = Config()

# 验证代码
if __name__ == "__main__":
    """验证配置是否正确加载"""
    print("验证配置加载...")
    print(f"MySQL 配置: {config.MYSQL_HOST}, {config.MYSQL_USER}, {config.MYSQL_DB}")
    print(f"MongoDB 配置: {config.MONGO_HOST}:{config.MONGO_PORT}, {config.MONGO_DB}")
    print(f"Milvus 配置: {config.MILVUS_HOST}:{config.MILVUS_PORT}, 集合: {config.MILVUS_COLLECTION_NAME}")
    print(f"Elasticsearch 配置: {config.ES_HOST}, 索引: {config.ES_INDEX_NAME}")
    print(f"模型路径: {config.MODEL_PATH}/{config.EMBEDDING_MODEL}, {config.RERANKER_MODEL}")
    print(f"本地简历目录: {config.LOCAL_RESUME_DIR}")
    print(f"临时文件目录: {config.TEMP_DIR}")
    print("配置验证通过！")