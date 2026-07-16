#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: router_A2Aagent_Server.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/3
描述: 
"""
from langchain_openai import ChatOpenAI
from python_a2a import run_server
from python_a2a.langchain import to_a2a_server
import asyncio
from agent_learn.config import Config

conf = Config()

async def main():
    # 创建LangChain LLM
    llm = ChatOpenAI(base_url=conf.base_url,
                     api_key=conf.api_key,
                     model=conf.model_name,
                     temperature=0.1,
                     streaming=True)
    # 转换为A2A服务器
    llm_server = to_a2a_server(llm)
    print(llm_server.agent_card)
    # 启动服务器
    run_server(llm_server, port=5555)

if __name__ == '__main__':
    asyncio.run(main())