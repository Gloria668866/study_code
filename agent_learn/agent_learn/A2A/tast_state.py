#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: tast_state.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from python_a2a import TaskState  # 只需相关导入
# 检查任务状态
if TaskState.COMPLETED == "completed":
    print("任务完成")
state = TaskState.SUBMITTED
print("转换后的状态值：", state.value)
print(state)