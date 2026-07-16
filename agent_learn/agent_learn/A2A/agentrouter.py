#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: agentrouter.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from python_a2a import AIAgentRouter, AgentNetwork
from langchain_openai import ChatOpenAI
from agent_learn.config import Config

conf=Config()

# 创建网络
network = AgentNetwork(name="MyNetwork")
network.add("TicketAgent", "http://127.0.0.1:5010")

# 创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# 创建路由器
router = AIAgentRouter(llm_client=llm, agent_network=network)
agent_name, confidence = router.route_query("预订票")
print(agent_name, confidence)