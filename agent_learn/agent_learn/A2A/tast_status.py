#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: tast_status.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from python_a2a import TaskStatus, TaskState

status_completed = TaskStatus(
    state=TaskState.COMPLETED,
    message={"info": "任务成功完成"}
)

status_failed = TaskStatus(
    state=TaskState.FAILED,
    message={"error": "无法处理请求"}
)

# 打印字典表示
print("完成状态：", status_completed.to_dict())
print("失败状态：", status_failed.to_dict())

# 完成状态： {'state': 'completed', 'timestamp': '2026-02-01T17:49:47.711505', 'message': {'info': '任务成功完成'}}
# 失败状态： {'state': 'failed', 'timestamp': '2026-02-01T17:49:47.711505', 'message': {'error': '无法处理请求'}}