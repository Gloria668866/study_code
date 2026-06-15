"""LLM 客户端：用 OpenAI 兼容协议，同时支持通义千问 Qwen 与 DeepSeek。
切换模型只需改 .env，不改代码。"""
import os

from openai import OpenAI
from .config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY

# 稳定性关键：上游 LLM 偶发挂起时，无超时 = 整个 /api/ask 永久转圈、连接被一直占着。
# timeout 让请求最多等 LLM_TIMEOUT 秒；max_retries 处理偶发网络抖动（指数退避）。
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# 惰性构造：OpenAI() 在 key 为空时构造即抛异常 → 原来没配 .env 连 import 都崩、/health 都起不来。
# 改为首次调用时才建客户端：服务能正常启动，只有真正用到 LLM 时才报清晰错误。
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY 未配置：请在 .env 中设置（参考 README 环境变量一节）")
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                         timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
    return _client


def chat(messages, temperature: float = 0.0, **kw) -> str:
    """同步对话，返回文本。Text2SQL 等确定性任务用 temperature=0。"""
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, **kw
    )
    return resp.choices[0].message.content or ""


def chat_stream(messages, temperature: float = 0.3, **kw):
    """流式对话，逐 token 产出（供洞察生成的 SSE 输出）。"""
    stream = _get_client().chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=temperature, stream=True, **kw
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
