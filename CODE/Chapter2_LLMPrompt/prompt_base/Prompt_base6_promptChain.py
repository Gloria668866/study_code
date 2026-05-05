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

# prompt chain
# step1
step1_prompt = """
你是传智教育的客服助手。推荐预算 2500 贝以内的课程套餐。课程包括：
- AI 大模型开发工程师（1000 贝，12 周，程序员）
- AI 大模型数据分析工程师（500 贝，10 周，数据分析师）
- AI 大模型运维工程师（500 贝，8 周，运维工程师）
- AI 大模型 Java 开发工程师（600 贝，15 周，Java 程序员）
输出课程列表。
"""

# 调用 DeepSeek API
step1_result = llm.invoke(step1_prompt)

# step2
step2_prompt = f"""
基于课程介绍：{step1_result.content}
推荐给预算 1500 贝的用户，输出 JSON：{{"推荐套餐": "名称", "理由": "说明"}}
"""
final_result = llm.invoke(step2_prompt)
print(final_result.content)


"""
D:\Anaconda\envs\llm_new\python.exe D:\LLM_Codes\Chapter2_LLMPrompt\Prompt_base6_promptChain.py 
```json
{
  "推荐套餐": "技术精简型",
  "理由": "在1500贝预算下，优先选择核心开发能力与关键应用场景课程，确保技术栈完整且性价比最高。推荐组合：AI 大模型开发工程师（1000贝） + AI 大模型数据分析工程师（500贝），覆盖开发与数据分析两大核心方向，剩余预算0贝。"
}
``` 

**可选替代方案**（如需调整方向）:
```json
{
  "推荐套餐": "运维速成型",
  "理由": "若用户侧重部署实践，可选择：AI 大模型运维工程师（500贝） + AI 大模型开发工程师（1000贝），剩余预算0贝。此组合适合快速掌握开发与运维闭环能力。"
}
```

Process finished with exit code 0

"""