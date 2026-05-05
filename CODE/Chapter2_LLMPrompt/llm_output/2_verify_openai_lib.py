# 2_verify_openai_lib.py
# 演示如何使用 openai 官方库调用大模型 API

import os
import json
from openai import OpenAI
import sys
import io

print("\n==================== 方法二：使用 `openai` 官方库 ====================")
# --- Step 1: 定义 API 配置并初始化客户端 ---
# 从环境变量读取 API 密钥，如果不存在则使用默认值
API_KEY ="YOUR_DEEPSEEK_API_KEY_HERE"
# API 的访问地址
API_URL = "https://api.deepseek.com/v1"
# 要使用的模型名称
MODEL = "deepseek-chat"

# 初始化 OpenAI 客户端
# 我们通过 base_url 参数将请求指向 DeepSeek 的 API 地址
client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL
)

# --- Step 2: 准备消息并调用 API ---
# 消息格式与直接 HTTP 请求中的 payload 类似
messages = [
    {"role": "system", "content": "你是一个由 DeepSeek 开发的乐于助人的人工智能助手。"},
    {"role": "user", "content": "你好！请用中文简单介绍一下你自己。JSON格式输出"}
# {"role": "user", "content": "中国的首都是哪里？ 首都： 请JSON输出"}
]

try:
    # 调用 chat.completions.create 方法发送请求
    # 参数与 HTTP 请求体中的字段一一对应
    response_object = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )

    # --- Step 3: 查看完整的响应对象 ---
    # 返回的是一个 ChatCompletion 对象，而不是纯字典
    print("\n--- 完整的 `openai` 库响应对象 (已转换为字典) ---")
    # 为了方便查看，我们使用 .model_dump() 方法将其转换为字典后打印
    print(json.dumps(response_object.model_dump(), indent=2, ensure_ascii=False))
    print("--------------------------------------------------")

    # --- Step 4: 从响应对象中提取关键信息 ---
    # 通过访问对象的属性来获取数据，代码更清晰，且有编辑器提示
    print("\n[解析结果]")

    # 获取模型回复内容
    model_content = response_object.choices[0].message.content
    print("模型回复内容转为dict: ",json.loads(model_content))
    # 获取 token 使用情况
    token_usage = response_object.usage

    print(f"模型回复内容: {model_content}")
    # token_usage 也是一个对象，可以直接打印查看
    print(f"Token 使用情况: {token_usage}")

except Exception as e:
    print(f"使用 `openai` 库时发生错误: {e}")
