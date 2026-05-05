# 3_1_verify_langchain.py
# 演示如何使用 langchain_openai 库调用大模型 API

import os
import json
from langchain_openai import ChatOpenAI
import sys
import io

print("\n==================== 方法三：使用 `langchain_openai` ====================")

# --- Step 1: 定义 API 配置并初始化 ChatOpenAI ---
API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE" #换成自己APIkey
# API 的访问地址
# API_URL = "https://api.deepseek.com/v1"
API_URL = "https://api.deepseek.com/v1"
# 要使用的模型名称
MODEL = "deepseek-chat"

# 初始化 ChatOpenAI
# 这是 LangChain 框架中与 OpenAI 兼容模型交互的核心类
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    temperature=0.7,
    max_tokens=150,
    streaming=  False
)

# --- Step 2: 准备消息并调用 API ---
# 消息格式与前两种方法一致
messages = [
    {"role": "system", "content": "你是一个由 DeepSeek 开发的乐于助人的人工智能助手。"},
    {"role": "user", "content": "你好！请用中文简单介绍一下你自己。"}
]


# 使用 .invoke() 方法是 LangChain 中推荐的、最直接的调用方式
response_message = llm.invoke(messages)
print(f"模型API接口响应内容: {response_message}")

# --- Step 3: 查看完整的响应对象 ---
# 返回的是一个 LangChain 的 AIMessage 对象
print("\n--- 完整的 `langchain` 响应对象 (已转换为字典) ---")
# AIMessage 对象没有 .dict() 或 .model_dump() 方法，我们手动构建字典来查看
response_dict = {
    "content": response_message.content, #大模型回复
    "response_metadata": response_message.response_metadata,
    "type": response_message.type,
    "id": response_message.id,
}
#美化输出
print(json.dumps(response_dict, indent=2, ensure_ascii=False))
print("--------------------------------------------------")

# --- Step 4: 从响应对象中提取关键信息 ---
# LangChain 对响应做了更高层次的抽象，提取信息非常直接
print("\n[解析结果]")

# 直接访问 .content 属性即可获得模型回复
model_content = response_message.content

# 其他元数据，包括 token 使用情况，存储在 .response_metadata 属性中
token_usage = response_message.response_metadata.get("token_usage", {})

print(f"模型回复内容: {model_content}")
print(f"Token 使用情况: {token_usage}")
