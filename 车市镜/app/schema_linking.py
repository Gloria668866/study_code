"""Schema Linking：表/列很多时，先挑出与问题相关的表，只把相关 schema 喂给 LLM。
本骨架用轻量关键词匹配；M2 可升级为向量检索（对表/列描述做 embedding）。"""
from .db import get_tables_meta, get_schema_text


def link_schema(question: str, max_tables: int = 6) -> str:
    meta = get_tables_meta()
    if len(meta) <= max_tables:
        return get_schema_text()  # 小库直接全量

    q = question.lower()
    scored = []
    for tbl, cols in meta.items():
        score = sum(tok in q for tok in [tbl.lower(), *(c.lower() for c in cols)])
        scored.append((score, tbl, cols))
    scored.sort(reverse=True)
    picked = [s for s in scored if s[0] > 0][:max_tables] or scored[:max_tables]
    return "\n".join(f"TABLE {t}({', '.join(cols)})" for _, t, cols in picked)
