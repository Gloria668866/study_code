# -*-coding:utf-8-*-
"""
RAGAS 评估脚本 — DeepSeek V4 做评估 LLM + 本地 BGE-M3 做 Embedding。
评估指标: faithfulness, answer_relevancy, context_precision, context_recall。

"""

import json, os, sys
from datetime import datetime

import pandas as pd
from openai import OpenAI
from datasets import Dataset

from ragas import evaluate
# RAGAS 0.4.x 中 legacy 指标（ragas.metrics._*）继承 Metric 基类，
# 可通过 evaluate() 验证；collections 指标基类不同，会报 TypeError。
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

# ---------- 项目路径 & 配置 ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from base import Config

conf = Config()

EVAL_JSON = os.path.join(current_dir, "rag_evaluate_data.json")
OUTPUT_DIR = os.path.join(current_dir, "results")
BGE_M3_PATH = os.path.join(rag_qa_path, "models", "bge-m3")


def build_llm():
    """ragas llm_factory + OpenAI 客户端 → DeepSeek V4，不经过 langchain。

    InstructorModelArgs 默认 max_tokens=1024，对 DeepSeek V4 推理模型太小
    （reasoning tokens 消耗大部分预算导致 JSON 输出截断）。
    llm_factory 的 adapter 内部硬编码了 InstructorModelArgs()，
    所以创建后手动提升 max_tokens。
    """
    client = OpenAI(
        api_key=conf.DEEPSEEK_API_KEY,
        base_url=conf.DEEPSEEK_BASE_URL,
    )
    llm = llm_factory(
        model=conf.LLM_MODEL,
        provider="openai",
        client=client,
    )
    llm.model_args["max_tokens"] = 8192
    return llm


def build_embeddings():
    """ragas 原生 HuggingFaceEmbeddings → 本地 BGE-M3，不经过 langchain。

    RAGAS 0.4.x 的 HuggingFaceEmbeddings 继承新的 BaseRagasEmbedding
    （提供 embed_text/embed_texts），但 legacy 指标需要旧 API
    BaseRagasEmbeddings（提供 embed_query/embed_documents）。
    这里用一个简单适配器做桥接。
    """
    raw = HuggingFaceEmbeddings(
        model=BGE_M3_PATH,
        device="cpu",
        normalize_embeddings=True,
    )
    return _LegacyEmbeddingsAdapter(raw)


class _LegacyEmbeddingsAdapter:
    """把 ragas 新 Embedding API（embed_text）适配为旧 API（embed_query）。"""

    def __init__(self, embedding):
        self._e = embedding

    def embed_query(self, text: str):
        return self._e.embed_text(text)

    def embed_documents(self, texts):
        return self._e.embed_texts(texts)

    async def aembed_query(self, text: str):
        return await self._e.aembed_text(text)

    async def aembed_documents(self, texts):
        return await self._e.aembed_texts(texts)


def load_eval_data():
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"加载评估数据: {len(data)} 条样本")
    return data


def to_ragas_dataset(data):
    return Dataset.from_dict({
        "question":      [item["question"]      for item in data],
        "answer":        [item["answer"]        for item in data],
        "contexts":      [item["context"]       for item in data],
        "ground_truth":  [item["ground_truth"]  for item in data],
    })


def save_results(result_df, scores):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(OUTPUT_DIR, f"ragas_scores_{ts}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"ragas_scores_{ts}.json")

    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {csv_path}")
    print(f"结果已保存: {json_path}")


def main():
    print("=" * 60)
    print("RAGAS 评估 — DeepSeek V4 + BGE-M3")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"LLM: {conf.LLM_MODEL}")
    print(f"Embedding: BGE-M3 ({BGE_M3_PATH})")
    print("=" * 60)

    # 1. 加载数据
    data = load_eval_data()
    dataset = to_ragas_dataset(data)

    # 2. 初始化评估组件
    print("\n初始化评估组件...")
    llm = build_llm()
    embeddings = build_embeddings()
    print("  LLM: DeepSeek V4 (ragas OpenAI provider)")
    print("  Embeddings: BGE-M3 (ragas HuggingFace provider)")

    # 3. 定义指标
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    # 4. 执行评估
    print("\n开始评估（共 %d 条样本）..." % len(data))
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    print("评估完成!")

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    result_df = result.to_pandas()
    print(result_df.to_string())

    # 汇总
    scores = {}
    print("\n各指标均值:")
    for col in result_df.columns:
        if col not in ("question", "answer", "contexts", "ground_truth"):
            try:
                avg = result_df[col].mean()
                scores[col] = round(float(avg), 4)
                print(f"  {col}: {avg:.4f}")
            except (TypeError, ValueError):
                pass

    # 6. 保存
    save_results(result_df, scores)


if __name__ == "__main__":
    main()
