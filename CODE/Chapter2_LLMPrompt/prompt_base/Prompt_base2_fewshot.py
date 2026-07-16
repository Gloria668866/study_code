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

# few-shot 提示
prompt = """
你是传智教育的客服助手。课程套餐包括：
- AI 大模型开发工程师（1000 贝，12 周，程序员）
- AI 大模型数据分析工程师（500 贝，10 周，数据分析师）
- AI 大模型运维工程师（500 贝，8 周，运维工程师）
- AI 大模型 Java 开发工程师（600 贝，15 周，Java 程序员）

示例1：
用户输入：预算 2000 贝以内。
输出：{"推荐套餐": "AI 大模型运维工程师", "理由": "价格 500 贝，符合预算。"}

示例2：
用户输入：我是 Java 程序员。
输出：{"推荐套餐": "AI 大模型 Java 开发工程师", "理由": "专为 Java 程序员设计。"}

用户输入：我是数据分析师，预算 3000 贝。
输出 JSON：{"推荐套餐": "名称", "理由": "说明"}
"""

# 调用 DeepSeek API
response = llm.invoke(prompt)
print(response.content)

"""
D:\Anaconda\envs\llm_new\python.exe D:\LLM_Codes\Chapter2_LLMPrompt\Prompt_base2_fewshot.py 
```json
{
  "推荐套餐": "AI 大模型数据分析工程师",
  "理由": "专为数据分析师设计，价格 500 贝，符合预算。"
}
```

Process finished with exit code 0

"""