# 3_1_verify_langchain.py
# 演示如何使用 langchain_openai 库以流式方式调用大模型 API
import os
from langchain_openai import ChatOpenAI
print("\n==================== 方法三：使用 `langchain_openai` (流式版) ====================")
# --- Step 1: 定义 API 配置并初始化 ChatOpenAI ---
API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE" #换成自己APIkey
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# 初始化 ChatOpenAI
# 核心改动 1: 将 streaming 设置为 True
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    temperature=0.7,
    max_tokens=150,
    streaming=True  # 启用流式输出
)

# --- Step 2: 准备消息并以流式方式调用 API ---
messages = [
    {"role": "system", "content": "你是一个由 DeepSeek 开发的乐于助人的人工智能助手。"},
    {"role": "user", "content": "你好！请用中文简单介绍一下你自己。JSON输出"}
]
print("\n[流式解析结果]")

try:
    # 核心改动 2: 使用 .stream() 方法代替 .invoke()
    # .stream() 返回一个数据块（chunk）的迭代器
    full_content = ""
    #初始化一个变量来存储最终的tokenusage 信息
    final_token_usage = {}
    for chunk in llm.stream(messages):
        # print("=========chunk========")
        # print(chunk)
        # chunk 是一个 AIMessageChunk 对象
        # 它的 .content 属性包含了本次接收到的文本片段
        content_piece = chunk.content
        if content_piece:
            # 实时打印文本片段，实现打字机效果
            print(content_piece, end="", flush=True) ##flush=True：立即刷新输出  #end=""：防止换行
            # 将文本片段累加到完整内容中
            full_content += content_piece

    # 流式结束后，打印一个换行符，让后续输出在新的一行
    print()

    # --- Step 3: 查看完整的响应内容 ---
    print("\n--- 流式传输完成后，聚合的完整内容 ---")
    print(full_content)
    print("--------------------------------------------------")

except Exception as e:
    print(f"\n使用 `langchain_openai` 库时发生错误: {e}")