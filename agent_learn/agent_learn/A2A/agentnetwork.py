#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agentnetwork.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from python_a2a import AgentNetwork
# 实例化一个 agentNetwork
network = AgentNetwork(name="MyNetwork")
# 添加一个agent，这个agent注册到了这个networt
network.add("TicketAgent", "http://127.0.0.1:5010")
# network.add("TicketAgent", "http://127.0.0.1:5009")

print(f"agent network-->{network.agent_cards}")
print('*'*80)

# 调用
client = network.get_agent("TicketAgent")
print(client.ask("预订一张从北京到上海的火车票"))