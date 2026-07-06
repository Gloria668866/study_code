#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: a2a_server.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

# 定义代理卡片
ticket_card = AgentCard(
    name="TicketAgentServer",
    description="票务代理",
    url="http://127.0.0.1:5009/a2a",
    skills=[AgentSkill(name="book_ticket", description="预订票务")]
)

# 自定义 A2AServer 子类
class TicketServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=ticket_card)

    def handle_task(self, task):
        print(f"任务状态：{task.status.state}")
        return task

# 启动服务器
if __name__ == "__main__":
    server = TicketServer()
    print(f"[{server.agent_card.name}] 创建服务成功")
    run_server(server, host="127.0.0.1", port=5009, debug=False)