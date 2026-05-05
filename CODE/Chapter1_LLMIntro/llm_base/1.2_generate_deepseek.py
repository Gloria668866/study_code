from langchain_openai import ChatOpenAI
# DeepSeek API 配置
API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"  # 替换为你的 DeepSeek API 密钥
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# 初始化 ChatOpenAI，配置 DeepSeek API
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    temperature=0.7,
    max_tokens=150
)

prompt = """
你是传智教育的客服助手。课程套餐包括：
- AI 大模型开发工程师（1000 贝，12 周，程序员）
- AI 大模型数据分析工程师（500 贝，10 周，数据分析师）
- AI 大模型运维工程师（500 贝，8 周，运维工程师）
- AI 大模型 Java 开发工程师（600 贝，15 周，Java 程序员）
根据用户输入，推荐合适的套餐，输出 JSON：{"推荐套餐": "名称", "理由": "说明"}。

用户输入：预算 2000 贝以内。
"""
# 调用 DeepSeek API
response = llm.invoke(prompt)
print(response.content)
