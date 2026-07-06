# SmartVoyage 配置文件
# 注意：请将 YOUR_API_KEY 替换为你自己的 API Key
# 支持的 API 提供商示例：
#   DeepSeek:   base_url='https://api.deepseek.com/v1', model='deepseek-chat'
#   SiliconFlow: base_url='https://api.siliconflow.cn/v1', model='Qwen/Qwen2.5-72B-Instruct'
#   OpenAI:     base_url='https://api.openai.com/v1', model='gpt-4o'

import os

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
env = "test"


class Config:

    def __init__(self):
        # LLM 配置
        self.base_url = 'https://api.deepseek.com/v1'
        self.api_key = 'YOUR_API_KEY'
        self.model_name = 'deepseek-chat'

        # MySQL 数据库配置
        self.host = 'localhost'
        self.user = 'root'
        self.password = 'YOUR_DB_PASSWORD'
        self.database = 'travel_rag'

        # 日志路径
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'app.log')

        # 12306 接口（暂未接入）
        self.url_123 = ""

        # 意图 → Agent 映射
        self.intent = {
            "weather": "WeatherQueryAssistant",
            "flight": "TicketQueryAssistant",
            "train": "TicketQueryAssistant",
            "concert": "TicketQueryAssistant",
            "order": "TicketOrderAssistant"
        }

        self.temperature = 0.1

    def get_mysql_config(self, env):
        if env == 'prod':
            self.host = 'localhost'
            self.user = 'root'
            self.password = 'YOUR_DB_PASSWORD'
            self.database = 'travel_rag'
        elif env == 'dev':
            self.host = 'localhost1'
            self.user = 'root1'
            self.password = 'YOUR_DB_PASSWORD'
            self.database = 'travel_rag'
        elif env == 'test':
            self.host = 'localhost2'
            self.user = 'root2'
            self.password = 'YOUR_DB_PASSWORD'
            self.database = 'travel_rag'
        else:
            self.host = 'localhost3'
            self.user = 'root3'
            self.password = 'YOUR_DB_PASSWORD'
            self.database = 'travel_rag'
        return self.host, self.user, self.password, self.database


if __name__ == '__main__':
    print(Config().log_file)
