# 1_verify_http.py
# 演示如何直接使用 HTTP 请求调用大模型 API
import os
import json
import requests
import sys
print("\n==================== 方法一：直接 HTTP 请求 ====================")

# --- Step 1: 定义 API 配置 ---
# 从环境变量读取 API 密钥，如果不存在则使用默认值
API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE" #换成自己的API密钥
# API 的访问地址
API_URL = "https://api.deepseek.com/chat/completions"
# 要使用的模型名称
MODEL = "deepseek-chat"

# --- Step 2: 准备请求头和请求体 ---
# 请求头包含认证信息和内容类型
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 请求体包含了所有需要发送给 API 的参数
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "你是一个由 DeepSeek 开发的乐于助人的人工智能助手。"},
        {"role": "user", "content": "你好！请用中文简单介绍一下你自己。给我两种句子，一个简单，一个详细。"}
    ],
    "temperature": 0.7, # 控制回复的随机性
    "max_tokens": 150,  # 控制回复的最大长度
    "stream": False,

}

# --- Step 3: 发送 POST 请求并获取响应 ---
# 使用 requests 库发送 POST 请求
# headers 包含了认证信息，json 参数会将 payload 字典自动转换为 JSON 字符串
http_response = requests.post(API_URL, headers=headers, json=payload)

# 检查 HTTP 状态码，如果不是 2xx 成功状态，则会抛出异常
if http_response.status_code != 200:
    print(f"HTTP 错误：{http_response.status_code}")
    print(f"错误信息：{http_response.text}")
    sys.exit(1)
# --- Step 4: 解析响应数据 ---
# API 返回的是 JSON 格式的字符串，我们使用 .json() 方法将其解析为 Python 字典
response_data = http_response.json()

print("\n--- 完整的 HTTP 响应 (已解析为字典) ---")
# 使用 json.dumps 美化打印输出的字典
print(json.dumps(response_data, indent=2, ensure_ascii=False))
print("-----------------------------------------")

# --- Step 5: 从解析后的数据中提取关键信息 ---
print("\n[解析结果]")

# 从字典中安全地获取模型回复内容
# .get("choices", [{}]) 确保即使答案choices 不存在也不会报错
model_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "未找到内容")

# 获取 token 使用情况
token_usage = response_data.get("usage", {})

print(f"模型回复内容: {model_content}")
print(f"Token 使用情况: {token_usage}")
