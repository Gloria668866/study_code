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

# COT 提示
prompt = """
你是传智教育的客服助手。课程套餐包括：
- AI 大模型开发工程师（1000 贝，12 周，程序员）
- AI 大模型数据分析工程师（500 贝，10 周，数据分析师）
- AI 大模型运维工程师（500 贝，8 周，运维工程师）
- AI 大模型 Java 开发工程师（600 贝，15 周，Java 程序员）

任务：分析预算 3000 贝以内、时长 10 周以上的套餐性价比（性价比=时长/价格）。
一步步思考：
1. 列出符合条件的套餐。
2. 计算性价比。
3. 推荐最高性价比套餐。
输出 JSON：{"推荐套餐": "名称", "推理过程": "步骤描述"}
"""


# 调用 DeepSeek API
response = llm.invoke(prompt)
print(response.content)

"""
D:\Anaconda\envs\llm_new\python.exe D:\LLM_Codes\Chapter2_LLMPrompt\Prompt_base3_COT.py 
```json
{
  "推荐套餐": "AI 大模型开发工程师",
  "推理过程": "1. 符合条件的套餐有：AI 大模型开发工程师（1000 贝，12 周）、AI 大模型数据分析工程师（500 贝，10 周）、AI 大模型 Java 开发工程师（600 贝，15 周）。2. 计算性价比：AI 大模型开发工程师（12/1000=0.012）、AI 大模型数据分析工程师（10/500=0.02）、AI 大模型 Java 开发工程师（15/600=0.025）。3. 最高性价比套餐是 AI 大模型 Java 开发工程师。"
}
```

Process finished with exit code 0

"""