#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: config.py
作者: ZZS
项目: LlmProject
创建日期: 2026/1/31
描述: 全局配置文件 - 使用前请替换 YOUR_API_KEY
"""

class Config:
    def __init__(self):
        # 推荐配置：DeepSeek / SiliconFlow / OpenAI 等兼容 API
        self.base_url = 'https://api.deepseek.com/v1'
        self.api_key = 'YOUR_API_KEY'
        self.model_name = 'deepseek-chat'
