#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: C02_ReActPattern.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from agent_learn.config import Config

conf = Config()

# 1.创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# 2.定义工具
# 关键修改：重写 multiply 工具，使其只接受一个字符串参数，并在内部解析。
@tool
def multiply(numbers_str: str) -> int:
    """用于计算两个整数的乘积。

    参数:
        numbers_str (str): 包含两个整数的字符串，用逗号分隔，例如："100,25"。
    返回:
        int: 两个整数的乘积。
    """
    print(f"正在执行乘法: {numbers_str}")
    try:
        a_str, b_str = numbers_str.split(',')
        a = int(a_str.strip())
        b = int(b_str.strip())
        return a * b
    except ValueError:
        return "输入的格式不正确，请确保是两个用逗号分隔的整数，例如：'100,25'"

@tool
def search_weather(city: str) -> str:
    """用于查询指定城市的实时天气。"""
    print(f"正在查询天气: {city}")
    if "北京" in city:
        return "北京今天是晴天，气温25摄氏度。"
    elif "上海" in city:
        return "上海今天是阴天，有小雨，气温22摄氏度。"
    else:
        return f"抱歉，我没有'{city}'的天气信息。"

# 将工具列表放入一个变量
tools = [multiply, search_weather]

# 3.自定义 ReAct 风格的 Prompt
react_prompt_template = """
角色：你是一个有用的 AI 助手，可以访问以下工具：
{tools}

任务：
请根据用户输入一步步推理，并按以下规则操作：
1. 每次输出只能包含一个动作（Action 和 Action Input）或一个最终答案（Final Answer）。
2. 如果用户输入包含多个任务，依次处理每个任务，不要一次性输出所有步骤。
3. 每次行动前，说明你的思考（Thought），并选择合适的工具或直接给出最终答案。
4. 如果需要使用工具，格式必须为：
   Thought: [你的思考]
   Action: [工具名称]
   Action Input: [工具的输入参数，例如对于multiply工具，使用'100,25'格式]
5. 如果可以直接回答或所有任务都完成，格式为：
   Thought: [你的思考]
   Final Answer: [最终答案]

可用的工具名称有: {tool_names}

用户输入: {input}
{agent_scratchpad}
"""

react_prompt = ChatPromptTemplate.from_template(react_prompt_template)

# 4.创建 ReAct 风格的 Agent
react_agent = create_react_agent(llm, tools, react_prompt)

# 5.创建 Agent Executor
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 启用错误处理，自动重试解析错误
)

# 6: 运行并测试 Agent
if __name__ == "__main__":
    # 测试用例1: 查询天气
    print("--- 运行Agent，查询: 上海今天的天气怎么样？ ---")
    response_weather = react_executor.invoke({"input": "上海今天的天气怎么样？"})
    print(f"\n--- Agent响应: ---")
    print(response_weather.get("output", "没有找到输出。"))
    print("-" * 30 + "\n")

    # 测试用例2: 数学计算
    # print("--- 运行Agent，查询: 100乘以25等于多少？ ---")
    # response_math = react_executor.invoke({"input": "100乘以25等于多少？"})
    # print(f"\n--- Agent响应: ---")
    # print(response_math.get("output", "没有找到输出。"))
    # print("-" * 30 + "\n")

    # 测试用例3: 包含多个任务
    # print("--- 运行Agent，查询: 100乘以25等于多少？ 上海的天气如何？ ---")
    # response_multi = react_executor.invoke({"input": "100乘以25等于多少？ 上海的天气如何？"})
    # print(f"\n--- Agent响应: ---")
    # print(response_multi.get("output", "没有找到输出。"))
    # print("-" * 30 + "\n")