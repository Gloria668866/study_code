#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: a2a_client_artifacts.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
import asyncio
from python_a2a import A2AClient

async def main():
    ticket_client = A2AClient("http://127.0.0.1:5010")

    #预订火车票
    ticket_query = "预订一张从北京到上海的火车票"
    print(f"[主控客户端日志]预订票务 -> '{ticket_query}'")
    ticket_result = ticket_client.ask(ticket_query)
    print(f"[主控客户端日志] 收到票务预订结果: {ticket_result}")

if __name__ == "__main__":
    asyncio.run(main())


# 收到A2A任务的task:=>
# Task(
#   id='1b39feef-bc01-4c95-90c6-5572eb1ac850',
#   session_id='13f0ea7a-9bce-4d89-8ce1-b57fcc73c2aa',
#   status=TaskStatus(state=<TaskState.SUBMITTED: 'submitted'>, message=None, timestamp='2026-02-01T17:58:32.589797'),
#   message={'content': {'text': '预订一张从北京到上海的火车票', 'type': 'text'}, 'role': 'user', 'message_id': 'fea16639-23ab-49c4-921e-a9b7566d2c34'},
#   history=[],
#   artifacts=[],
#   metadata={}
# )

# 输出结果task:
# Task(
#   id='1b39feef-bc01-4c95-90c6-5572eb1ac850',
#   session_id='13f0ea7a-9bce-4d89-8ce1-b57fcc73c2aa',
#   status=TaskStatus(state=<TaskState.COMPLETED: 'completed'>, message=None, timestamp='2026-02-01T17:58:32.605771'),
#   message={'content': {'text': '预订一张从北京到上海的火车票', 'type': 'text'}, 'role': 'user', 'message_id': 'fea16639-23ab-49c4-921e-a9b7566d2c34'},
#   history=[],
#   artifacts=[{'parts': [{'type': 'text', 'text': '上海到北京的火车票已经预订成功！  G1001,10车1A '}]}],
#   metadata={}
# )