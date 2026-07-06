# 导入 BM25 算法
from rank_bm25 import BM25Okapi
import numpy as np
import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
module_dir = os.path.dirname(current_dir)
# print(f'module_dir--》{module_dir}')
sys.path.insert(0, module_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)
# 系统配置

from mysql_qa.utils.preprocess import preprocess_text
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.cache.redis_client import RedisClient
# 导入日志
from base import logger


class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()

    #单下划线 (_)：表示“不建议”外部修改或调用（软性的保护）
    def _load_data(self):
        original_key ="qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从redis中获取原始问题（快）
        self.original_questions = self.redis_client.get_data(original_key)
        # 从redis中获取分词后的问题（快）
        tokenized_questions  = self.redis_client.get_data(tokenized_key)

        if not self.original_questions or not tokenized_questions:
            # 从Mysql中获取问题 如果redis中没有数据
            self.original_questions=self.mysql_client.fetch_questions()
            if not self.original_questions:
                self.logger.info("未加载到数据")
                return
            #分词问题
            tokenized_questions = [preprocess_text(question[0]) for question in self.original_questions]
            #print("tokenized_questions--->", tokenized_questions)
            #存储原始问题到redis
            self.redis_client.set_data(original_key, [(question[0]) for question in self.original_questions])
            #存储分词后问题到redis
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # 设置问题列表
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录 BM25 初始化成功
        self.logger.info("BM25 模型初始化完成")

    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()


    def _question_text(self, question):
        if isinstance(question, (list, tuple)):
            return question[0] if question else ""
        return question

    def search_with_meta(self, query, threshold=0.85):
        """搜索查询并返回可观测元数据。

        Returns:
            tuple: (answer, need_rag, meta)
        """
        meta = {
            "cache_hit": False,
            "bm25_score": None,
            "bm25_threshold": threshold,
            "best_question": "",
            "route": "bm25",
        }
        if not query or not isinstance(query, str):
            #记录无效查询
            self.logger.error("无效查询")
            meta["route"] = "invalid_query"
            return None, False, meta
        # 检查Redis缓存
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            meta["cache_hit"] = True
            meta["route"] = "redis_cache"
            return cached_answer, False, meta

        try:
            if not self.bm25 or not self.questions:
                self.logger.warning("BM25 模型未初始化，回退到 RAG")
                meta["route"] = "bm25_unavailable"
                return None, True, meta
            # 分词查询
            query_tokens = preprocess_text(query)
            # 计算 BM25 分数
            scores = self.bm25.get_scores(query_tokens)
            # 计算 Softmax 分数
            softmax_scores = self._softmax(scores)
            # 获取最高分索引
            best_idx = softmax_scores.argmax()
            # 获取最高分
            best_score = softmax_scores[best_idx]
            meta["bm25_score"] = round(float(best_score), 4)
            meta["best_question"] = self._question_text(self.original_questions[best_idx])
            # 检查是否超过阈值
            if best_score >= threshold:
                # 获取原始问题
                original_question = self._question_text(self.original_questions[best_idx])
                # 获取答案
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 缓存答案
                    self.redis_client.set_data(f"answer:{query}", answer)
                    # 记录搜索成功
                    self.logger.info(f"搜索成功，Softmax 相似度: {best_score:.3f}")
                    # 返回答案和 False
                    meta["route"] = "mysql_bm25"
                    return answer, False, meta
            # 记录无可靠答案
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            # 返回 None 和 True
            meta["route"] = "bm25_miss"
            return None, True, meta
        except Exception as e:
            # 记录搜索失败
            self.logger.error(f"搜索失败: {e}")
            # 返回 None 和 True
            meta["route"] = "bm25_error"
            meta["error"] = str(e)
            return None, True, meta

    def search(self, query, threshold=0.85):
        #搜索查询
        answer, need_rag, _ = self.search_with_meta(query, threshold=threshold)
        return answer, need_rag

if __name__ == '__main__':
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)




