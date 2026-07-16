#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: main_orchestrator.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/3
描述: 
"""
import asyncio
from python_a2a import AgentNetwork, A2AClient, Task, Message, MessageRole, TextContent
import json
import uuid
from time import sleep


async def main():
    # 步骤1：初始化 AgentNetwork 并注册专家 Agent
    # 1.1 创建 AgentNetwork 实例，用于管理 Agent 集合
    network = AgentNetwork(name="TravelOrchestrator")
    # 1.2 添加票务 Agent
    network.add("TicketAgent", "http://127.0.0.1:5009")
    # 1.3 添加天气 Agent
    network.add("WeatherAgent", "http://127.0.0.1:5008")
    print("[主控日志] AgentNetwork 初始化完成，已添加专家代理：")
    for agent_info in network.list_agents():
        print(json.dumps(agent_info, indent=4, ensure_ascii=False))
    print("-" * 50)

    # 步骤2：执行串行任务流
    # 2.1 任务一：查询天气
    weather_query = "北京的天气怎么样"  # 用户请求
    print(f"[主控日志] 串行任务第一步：向 WeatherAgent 查询天气 -> '{weather_query}'")
    # 获取 WeatherAgent 客户端
    weather_client = network.get_agent("WeatherAgent")
    # 创建消息和任务
    message_weather = Message(content=TextContent(text=weather_query), role=MessageRole.USER)
    task_weather = Task(id="task-" + str(uuid.uuid4()), message=message_weather.to_dict())

    # 发送任务并等待结果，这是串行执行的关键
    weather_result_task = await weather_client.send_task_async(task_weather)
    sleep(1)  # 模拟处理延迟

    # 从返回的任务中解析出天气信息
    weather_info = "未知天气"
    try:
        # 获取 artifacts 中的文本部分
        weather_parts = weather_result_task.artifacts[0]["parts"]
        if weather_parts and weather_parts[0].get("type") == "text":
            weather_info = weather_parts[0].get("text")
            print(f"[主控日志] 收到 WeatherAgent 的结果: '{weather_info}'")
    except Exception as e:
        print(f"[主控日志] 解析天气结果出错: {e}")

    # 2.2 任务二：根据天气结果预订火车票
    # 这是将一个 Agent 的输出作为另一个 Agent 的输入的关键步骤
    print(f"\n[主控日志] 串行任务第二步：根据天气结果决定预订火车票")
    ticket_query = f"预订一张从北京到上海的火车票，当前天气是：{weather_info}"
    print(f"[主控日志] 传递给 TicketAgent 的查询为: '{ticket_query}'")
    # 获取 TicketAgent 客户端
    ticket_client = network.get_agent("TicketAgent")
    # 创建新的任务
    message_ticket = Message(content=TextContent(text=ticket_query), role=MessageRole.USER)
    task_ticket = Task(id="task-" + str(uuid.uuid4()), message=message_ticket.to_dict())

    # 发送任务并获取最终结果
    ticket_result_task = await ticket_client.send_task_async(task_ticket)
    sleep(1)  # 模拟处理延迟

    # 打印最终结果
    print(f"\n[主控日志] 收到 TicketAgent 的最终结果：")
    print(json.dumps(ticket_result_task.to_dict(), indent=4, ensure_ascii=False))
    print("-" * 50)

    print("[主控日志] 所有串行任务完成！")


if __name__ == "__main__":
    # 请确保 weather_agent.py 和 ticket_agent.py 正在运行...
    asyncio.run(main())