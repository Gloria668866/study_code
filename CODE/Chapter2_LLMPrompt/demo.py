from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

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
from langchain.tools import tool

# 模拟的课程数据库
course_database = [
    {"name": "AI 大模型开发工程师", "price": 3000, "duration_weeks": 12, "target": "程序员"},
    {"name": "AI 大模型数据分析工程师", "price": 2500, "duration_weeks": 10, "target": "数据分析师"},
    {"name": "AI 大模型运维工程师", "price": 2000, "duration_weeks": 8, "target": "运维工程师"},
    {"name": "AI 大模型 Java 开发工程师", "price": 3500, "duration_weeks": 15, "target": "Java 程序员"}
]

@tool
def query_courses(occupation: str, max_budget: int) -> list:
    """
    根据职业和最高预算查询符合条件的课程。
    """
    results = []
    for course in course_database:
        # 职业匹配（包含关系，例如'程序员'能匹配'Java 程序员'）
        if occupation in course["target"] and course["price"] <= max_budget:
            results.append(course)
    return results
# 定义 agent 可以使用的工具列表
tools = [query_courses]

# 从 LangChain Hub 获取标准的 ReAct 提示模板
# 这个模板已经包含了指导 LLM 如何进行思考和行动的指令
prompt_template = hub.pull("hwchase17/react")
print("--- [客户端日志: 获取标准 ReAct 提示模板] ---")
print(prompt_template)

# 创建 ReAct Agent
agent = create_react_agent(llm, tools, prompt_template)

# 创建 Agent 执行器，它负责运行整个 ReAct 循环
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True 可以打印出模型的思考过程

# 调用 Agent！
user_input = "我是程序员，预算 2500 贝，帮我推荐课程"
response = agent_executor.invoke({"input": user_input})

print("\n--- 最终回答 ---")
print(response["output"])