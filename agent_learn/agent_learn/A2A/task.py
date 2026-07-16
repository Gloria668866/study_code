#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: task.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from python_a2a import Task, Message, MessageRole, TextContent

# 创建任务
message = Message(content=TextContent(text="查询天气"), role=MessageRole.USER)
task = Task(message=message.to_dict())
print(task)

# Task(
#   id='a50f2381-e890-4e89-b461-c913f3cd4ccb',
#   session_id='0bfaea44-5118-4009-897a-09f2a5d74712',
#   status=TaskStatus(state=<TaskState.SUBMITTED: 'submitted'>, message=None, timestamp='2026-02-01T17:42:10.886472'),
#   message={'content': {'text': '查询天气', 'type': <ContentType.TEXT: 'text'>}, 'role': 'user', 'message_id': 'cf3216e8-e49d-4d28-8c24-0df32b08422f'},
#   history=[],
#   artifacts=[], # 客户端把任务task给到服务端，服务端完成任务后需要把结果放到artifacts里面。
#   metadata={}
# )