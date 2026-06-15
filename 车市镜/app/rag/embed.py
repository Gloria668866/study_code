"""BGE-large-zh 向量化（本地，1024 维）。

要点（PRD-2 §5.3/§5.4/§5.8）：
- query/passage **非对称**：检索侧 query 要加指令前缀，入库的 passage **不加前缀**；加错召回明显掉。
- **归一化**（normalize）后用余弦相似度（pgvector vector_cosine_ops / HNSW 同款）。
- **512 token 硬上限**：超了被截断丢语义；token 用模型自带 tokenizer 精确数（中文按 token 不按字符）。
- **版本一致性**：入库与检索必须同一模型同一版本同样归一化，否则向量空间不一致、召回全乱（§5.8）。
  模型名/版本在 config（EMBED_MODEL_NAME/VERSION）；换模型必须全量重算、重建 HNSW。
"""
import os

from ..config import EMBED_MODEL_NAME, EMBED_MAX_TOKENS, RERANK_MODEL_NAME

# bge-*-zh-v1.5 检索 query 的指令前缀（passage 不加）
_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_model = None
_reranker = "unloaded"     # "unloaded" / None(不可用) / CrossEncoder 实例


def get_model():
    """懒加载单例（模型重，首次会下载 ~1.3GB 权重）。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
        _model.max_seq_length = EMBED_MAX_TOKENS
    return _model


def count_tokens(text: str) -> int:
    """用模型自带 tokenizer 数 token（保证切块 ≤512 的依据）。"""
    tok = get_model().tokenizer
    return len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])


def embed_passages(texts):
    """入库侧：passage 不加前缀、归一化。返回 list[list[float]]（1024 维）。"""
    if not texts:
        return []
    vecs = get_model().encode(texts, normalize_embeddings=True,
                              batch_size=32, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def embed_query(text: str):
    """检索侧：query 加指令前缀、归一化。返回 list[float]。"""
    vec = get_model().encode([_QUERY_INSTRUCTION + text], normalize_embeddings=True)[0]
    return vec.tolist()


def _get_reranker():
    """懒加载 bge-reranker（CrossEncoder）；模型不可用返回 None（调用方降级用 RRF 分）。"""
    global _reranker
    if _reranker == "unloaded":
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANK_MODEL_NAME, max_length=512)
        except Exception as e:               # 模型没下到/加载失败 → 降级
            print(f"[warn] bge-reranker 不可用，降级为 RRF 排序：{e}")
            _reranker = None
    return _reranker


def rerank_scores(query: str, passages):
    """对 (query, passage) 精排，返回 [0,1] 相关分。reranker 不可用 → None。
    （sentence-transformers 的 CrossEncoder 对单标签模型默认已套 sigmoid，输出即 0~1，不再二次 sigmoid。）"""
    rr = _get_reranker()
    if rr is None or not passages:
        return None
    scores = rr.predict([(query, p) for p in passages])
    return [float(s) for s in scores]
