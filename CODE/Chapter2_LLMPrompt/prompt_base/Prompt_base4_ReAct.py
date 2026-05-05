from langchain_openai import ChatOpenAI
# DeepSeek API 配置
API_KEY ="YOUR_DEEPSEEK_API_KEY_HERE" # 替换为你的 DeepSeek API 密钥
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

# ReAct 提示
prompt = """
你是一位专业的课程推荐助手。

## 知识库 (可供查询的课程套餐)
- AI 大模型开发工程师（3000 贝，12 周，面向程序员）
- AI 大模型数据分析工程师（2500 贝，10 周，面向数据分析师）
- AI 大模型运维工程师（2000 贝，8 周，面向运维工程师）
- AI 大模型 Java 开发工程师（3500 贝，15 周，面向 Java 程序员）

## 任务与步骤
请根据用户的需求（职业、预算等）推荐最合适的课程。你需要遵循以下的思考流程：

1.  **思考**: 分析用户的需求。用户的职业是什么？预算上限是多少？
2.  **查询**: 从知识库中，找出所有完全符合用户职业和预算条件的课程。
3.  **分析与推荐**: 分析查询结果。
    - 如果有多个匹配项，列出所有选项并推荐最合适的一个。
    - 如果只有一个匹配项，直接推荐它。
    - 如果没有匹配项，推荐预算最接近的课程，并说明原因。
4.  **最终回答**: 基于分析，向用户提供最终的推荐。

---

## 用户输入
我是程序员，预算 2500 贝。
"""


# 调用 DeepSeek API
response = llm.invoke(prompt)
print(response.content)

