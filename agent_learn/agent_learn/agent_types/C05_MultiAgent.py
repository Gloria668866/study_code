#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: C05_MultiAgent.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate  # 导入所有必需的 Prompt 类
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import StrOutputParser
from agent_learn.config import Config

conf = Config()

# 1.创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)


# 2.定义工具
# 2.1 计算工具
@tool
def multiply(a: int, b: int) -> int:
    """用于计算两个整数的乘积。

    参数:
        a (int): 第一个整数。
        b (int): 第二个整数。
    """
    print(f"\n[计算专家] -> 正在执行乘法: {a} * {b}")
    return a * b


@tool
def add(a: int, b: int) -> int:
    """用于计算两个整数的和。

    参数:
        a (int): 第一个整数。
        b (int): 第二个整数。
    """
    print(f"\n[计算专家] -> 正在执行加法: {a} + {b}")
    return a + b


# 2.2 信息查询工具
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


@tool
def get_current_date() -> str:
    """用于获取当前日期。"""
    print("\n[信息专家] -> 正在获取当前日期...")
    import datetime
    return datetime.date.today().strftime("%Y年%m月%d日")


# 3 创建两个独立的 Agent
# 3.1 创建“计算专家” Agent
math_tools = [multiply, add]
# 创建完整的 Tool Calling Prompt
# 这包括一个系统消息，一个用户消息占位符，以及一个 Agent 中间思考过程的占位符。
math_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个强大的数学计算专家，可以访问和使用各种数学工具。"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
# 创建 math_Agent
math_agent = create_tool_calling_agent(llm, math_tools, math_prompt)
# 创建 math Agent Executor
math_executor = AgentExecutor(
    agent=math_agent,
    tools=math_tools,
    verbose=True
)

# 3.2 创建“信息专家” Agent
info_tools = [search_weather, get_current_date]
# 手动创建完整的 Tool Calling Prompt
info_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个强大的信息查询专家，可以访问和使用各种查询工具。"),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
# 创建 info Agent
info_agent = create_tool_calling_agent(llm, info_tools, info_prompt)
# 创建 info Agent Executor
info_executor = AgentExecutor(
    agent=info_agent,
    tools=info_tools,
    verbose=True
)

# 4.协调和总结工作流
def multi_agent_workflow(query: str, math_task: str, info_task: str):
    print("--- 启动多智能体协作流程 ---")
    print(f"\n用户原始请求: {query}")

    # 4.1 让“计算专家”执行任务
    print("\n[主程序] -> 将任务分配给计算专家...")
    math_result = math_executor.invoke({"input": math_task}).get("output")
    print(f"\n[主程序] -> 计算专家返回结果: {math_result}")

    # 4.2 让“信息专家”执行任务
    print("\n[主程序] -> 将任务分配给信息专家...")
    info_result = info_executor.invoke({"input": info_task}).get("output")
    print(f"\n[主程序] -> 信息专家返回结果: {info_result}")

    # 4.3 使用 LLM 进行最终结果总结
    print("\n[主程序] -> 使用大模型进行最终总结...")
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个善于总结和整合信息的助手。请根据以下信息，为用户原始请求生成一个完整的回答。"),
        ("human",
         f"用户请求: {query}\n\n计算结果: {math_result}\n\n信息查询结果: {info_result}\n\n请整合以上信息，生成一个连贯的最终回答。")
    ])
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    final_answer = summarize_chain.invoke({"query": query})

    print("\n--- 协作流程已完成！---")
    print(f"最终综合回答:\n{final_answer}")
    return final_answer


if __name__ == "__main__":
    # 定义用户原始请求和分配给每个Agent的子任务
    original_query = "请先计算 25 乘以 4，然后告诉我北京今天的天气和当前日期。"
    math_task = "计算 25 乘以 4"
    info_task = "查询北京今天的天气和当前日期"

    # 启动工作流
    multi_agent_workflow(original_query, math_task, info_task)