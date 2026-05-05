from langchain_openai import ChatOpenAI
# DeepSeek API 配置
API_KEY ="YOUR_DEEPSEEK_API_KEY_HERE"  # 替换为你的 DeepSeek API 密钥
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

# reflection 提示(适合交互任务)
# reflection：生成初步答案后反思修改，提高准确性。
prompt = """
你是传智教育的客服助手。推荐预算 2500 贝以内的课程套餐。课程包括：
- AI 大模型开发工程师（1000 贝，12 周，程序员）
- AI 大模型数据分析工程师（500 贝，10 周，数据分析师）
- AI 大模型运维工程师（500 贝，8 周，运维工程师）
- AI 大模型 Java 开发工程师（600 贝，15 周，Java 程序员）
输出 JSON：{"推荐套餐": "名称", "理由": "说明"}
"""

# 调用 DeepSeek API
initial_result = llm.invoke(prompt)
print(initial_result.content)
# 反思
reflection_prompt = f"""
初步答案：{initial_result.content}
反思：检查是否遗漏预算或适用人群限制，是否推荐最优套餐。输出优化后的 JSON：{{"推荐套餐": "名称", "理由": "说明"}}
"""
final_result = llm.invoke(reflection_prompt)
print(final_result.content)









