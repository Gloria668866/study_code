#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: C01_define_tool.py
作者: ZZS
项目: LlmProject
创建日期: 2026/1/31
描述: 
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from agent_learn.config import Config

conf = Config()


# todo: 第一步：定义工具函数
def add(a: int, b: int) -> int:
    """
    将数字a与数字b相加
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """
    将数字a与数字b相乘
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a * b


# 定义 JSON 格式的工具 schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "将数字a与数字b相加",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "将数字a与数字b相乘",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]


# todo: 第二步：初始化模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)
# 绑定工具，允许模型自动选择工具
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

# todo: 第三步：调用回复
query = "2+1等于多少？"
messages = [HumanMessage(query)]

try:
    # todo: 第一次调用
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    print(f"\n第一轮调用后结果：\n{messages}")

    # 处理工具调用
    # 判断消息中是否有tool_calls，以判断工具是否被调用
    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            # todo: 处理工具调用
            selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
            tool_output = selected_tool(**tool_call["args"])
            messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
        print(f"\n第二轮  message中增加tool_output 之后：\n{messages}")

        # todo: 第二次调用，将工具结果传回模型以生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n最终模型响应：\n{final_response.content}")
    else:
        print("模型未生成工具调用，直接返回文本:")
        print(ai_msg.content)
except Exception as e:
    print(f"模型调用失败: {str(e)}")

# 第一轮调用后结果：
# [HumanMessage(content='2+1等于多少？', additional_kwargs={}, response_metadata={}), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '019c1221ba9ae39f2bb455df7504e99a', 'function': {'arguments': '{"a": 2, "b": 1}', 'name': 'add'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 290, 'total_tokens': 316, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'Qwen/Qwen2.5-72B-Instruct', 'system_fingerprint': '', 'id': '019c1221b68b26a9c598a74265788d34', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--dfadd979-8912-4ac2-93cb-5dcbe457837d-0', tool_calls=[{'name': 'add', 'args': {'a': 2, 'b': 1}, 'id': '019c1221ba9ae39f2bb455df7504e99a', 'type': 'tool_call'}], usage_metadata={'input_tokens': 290, 'output_tokens': 26, 'total_tokens': 316, 'input_token_details': {}, 'output_token_details': {}})]

# 第二轮  message中增加tool_output 之后：
# [HumanMessage(content='2+1等于多少？', additional_kwargs={}, response_metadata={}), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '019c1221ba9ae39f2bb455df7504e99a', 'function': {'arguments': '{"a": 2, "b": 1}', 'name': 'add'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 290, 'total_tokens': 316, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'Qwen/Qwen2.5-72B-Instruct', 'system_fingerprint': '', 'id': '019c1221b68b26a9c598a74265788d34', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--dfadd979-8912-4ac2-93cb-5dcbe457837d-0', tool_calls=[{'name': 'add', 'args': {'a': 2, 'b': 1}, 'id': '019c1221ba9ae39f2bb455df7504e99a', 'type': 'tool_call'}], usage_metadata={'input_tokens': 290, 'output_tokens': 26, 'total_tokens': 316, 'input_token_details': {}, 'output_token_details': {}}), ToolMessage(content='3', tool_call_id='019c1221ba9ae39f2bb455df7504e99a')]

# 最终模型响应：
# 2+1等于3。
