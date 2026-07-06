# router_A2Aagent_Server.py
# 功能：路由Agent服务器，使用LLM进行意图识别和路由决策，支持工作流。

import logging
from python_a2a import run_server, Message, MessageRole, TextContent
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from python_a2a import run_server, AgentCard
from python_a2a import A2AClient
from python_a2a.langchain import to_a2a_server
print(f"Using real ChatOpenAI model with API key")
import asyncio
import os
import sys
sys.path.append("//utils")
import threading
import time
import socket
from typing import Dict, List, Any
from Chapter4_Agent.SmartVoyage.config import Config
conf=Config()
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 创建LangChain LLM
    llm = ChatOpenAI(
        model=conf.model_name,
        api_key=conf.api_key,
        base_url=conf.api_url,
        temperature=0,
        streaming=True
    )

    # 转换为A2A服务器
    llm_server = to_a2a_server(llm)
    print(llm_server.agent_card)
    # 启动服务器
    run_server(llm_server, port=6666)

if __name__ == '__main__':
    asyncio.run(main())