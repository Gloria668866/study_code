#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: weather_agent.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/3
描述: 
"""
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

# A2A Agent 的名片
agent_card = AgentCard(
    name="WeatherAgentServer",
    description="一个天气预报查询的专家 Agent",
    url="http://127.0.0.1:5008",
    skills=[AgentSkill(name="query", description="接受天气查询查询",examples=["天气北京"])]
)


class WeatherAgentServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=agent_card)

    def handle_task(self, task):
        print("收到A2A任务的task:=>", task)
        query = (task.message or {}).get("content", {}).get("text", "")
        print(f"[{self.agent_card.name} 日志] 收到 A2A 任务: '{query}'")
        # 决策：如果查询包含“天气”，就调用 MCP 工具
        if "天气" in query:
            print(f"[{self.agent_card.name} 日志] 决策：任务需要天气数据，准备调用工具...")
            try:
                # 这里的结果可以来自于 MCP 模块，这里我们直接模拟结果
                weather_result = {"温度": 30, "天气": "晴天"}
                print(f"[{self.agent_card.name} 日志] 从 MCP 工具获得结果: '{weather_result}'")
                # 将结果保存为任务 artifacts,artifacts是任务的输出结果
                task.artifacts = [{"parts": [{"type": "text", "text": weather_result}]}]
            except Exception as e:
                error_msg = f"调用 工具失败: {e}"
                print(f"[{self.agent_card.name} 日志] {error_msg}")
                task.artifacts = [{"parts": [{"type": "text", "text": error_msg}]}]
        else:
            task.artifacts = [{"parts": [{"type": "text", "text": "无法理解的任务"}]}]

        task.status = TaskStatus(state=TaskState.COMPLETED)
        print(f"[{self.agent_card.name} 日志] 任务处理完毕")
        print(f"[{self.agent_card.name} 日志] 输出结果task: {task}")
        print(f"[{self.agent_card.name} 日志] 输出结果task.artifacts: {task.artifacts}")
        return task


if __name__ == "__main__":
    server = WeatherAgentServer()
    print(f"[{server.agent_card.name}] 已启动，在 {server.agent_card.url}")
    run_server(server, host="127.0.0.1", port=5008)