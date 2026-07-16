#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: main.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/3
描述: 
"""
import asyncio
from python_a2a import AgentNetwork, AIAgentRouter, A2AClient, Task, Message, MessageRole, TextContent
import json
import uuid

from time import sleep


async def main():
    # 1. 创建 AgentNetwork
    network = AgentNetwork(name="TravelAgentNetwork")
    # 添加票务代理
    network.add("TicketAgent", "http://127.0.0.1:5009")
    # 添加天气代理
    network.add("WeatherAgent", "http://127.0.0.1:5008")
    print("[主控日志] AgentNetwork 初始化完成，已添加代理：")
    for agent_info in network.list_agents():
        print(json.dumps(agent_info, indent=4, ensure_ascii=False))
    print("-" * 50)

    # 2. 创建 AIAgentRouter
    router=AIAgentRouter(
        llm_client=A2AClient("http://localhost:5555"),  # 假设使用router_A2Aagent_Server.py的LLM服务器
        agent_network=network)

    # 3. 测试查询
    queries = [
        "帮我查下北京的天气",  # 应该路由到 WeatherAgent
        "预订一张从北京到上海的火车票"  # 应该路由到 TicketAgent
    ]
    for query in queries:
        print(f"[主控日志] 用户查询: '{query}'")
        # 使用路由器选择代理
        agent_name, confidence = router.route_query(query)
        print(f"[主控日志] 路由结果: {agent_name} (置信度: {confidence})")
        if agent_name:
            # 获取代理客户端
            agent_client = network.get_agent(agent_name)
            if agent_client:
                # 创建消息和任务
                message = Message(
                    content=TextContent(text=query),
                    role=MessageRole.USER
                )
                task = Task(
                    id="task-" + str(uuid.uuid4()),
                    message=message.to_dict())

                # 发送任务并获取响应
                try:
                    result = await agent_client.send_task_async(task)
                    sleep(5)
                    print("[主控日志] 收到完整响应：")
                    print(json.dumps(result.to_dict(), indent=4, ensure_ascii=False))
                except:
                    print("[主控日志] 错误：无法连接到代理")


if __name__ == "__main__":
    # 请确保 ticket_agent.py 和 weather_agent.py 正在运行...
    asyncio.run(main())