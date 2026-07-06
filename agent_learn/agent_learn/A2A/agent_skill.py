#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agent_skill.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from python_a2a import  AgentSkill
# 定义一个代理技能
ticket_skill = AgentSkill(
    name="book_ticket",
    description="预订火车票的技能",
    examples=["预订从上海到北京的火车票"],
    input_modes=["text/plain"],  # text/html
    output_modes=["text/plain"]
)

print(ticket_skill)
print(ticket_skill.to_dict())

# AgentSkill(name='book_ticket', description='预订火车票的技能', id='f616bf0b-c86a-4492-bcd8-5c111c3cdd2b', tags=[], examples=['预订从上海到北京的火车票'], input_modes=['text/plain'], output_modes=['text/plain'])
# {'id': 'f616bf0b-c86a-4492-bcd8-5c111c3cdd2b', 'name': 'book_ticket', 'description': '预订火车票的技能', 'tags': [], 'examples': ['预订从上海到北京的火车票'], 'inputModes': ['text/plain'], 'outputModes': ['text/plain']}