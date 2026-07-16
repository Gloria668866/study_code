#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: C03_pydantic.py
作者: ZZS
项目: LlmProject
创建日期: 2026/1/31
描述: 
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
"""
Pydantic 是一个 Python 库，用于数据验证和序列化。
它通过使用 Python 类型注解（type hints）来定义数据模型，
并提供强大的数据验证功能。Pydantic 基于 Python 的 dataclasses 和 typing 模块，
允许开发者定义结构化的数据模型，并自动验证输入数据是否符合指定的类型和约束。
"""
from pydantic.v1 import BaseModel, Field

from agent_learn.config import Config

conf = Config()



# todo: 第一步：定义工具函数
class Add(BaseModel):
    """
    将两个数字相加
    """
    a: int = Field(..., description="第一个数字")
    b: int = Field(..., description="第二个数字")

    def invoke(self, args):
        # 验证参数
        tool_instance = self.__class__(**args)  # 自动验证 a 和 b
        return tool_instance.a + tool_instance.b

class Multiply(BaseModel):
    """
    将两个数字相乘
    """
    a: int = Field(..., description="第一个数字")
    b: int = Field(..., description="第二个数字")

    def invoke(self, args):
        # 验证参数
        tool_instance = self.__class__(**args)  # 自动验证 a 和 b
        return tool_instance.a * tool_instance.b

# 定义 JSON 格式的工具 schema
tools = [Add, Multiply]


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
            selected_tool = {"add": Add, "multiply": Multiply}[tool_call["name"].lower()]
            # 实例化工具类并调用 invoke
            tool_instance = selected_tool(**tool_call["args"])
            tool_output = tool_instance.invoke(tool_call["args"])
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