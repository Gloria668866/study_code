import os
from dotenv import load_dotenv

from common.path_utils import get_file_path

# 优先加载项目根目录下的 .env，兼容本地开发
load_dotenv(get_file_path(".env"))


class Config:
    def __init__(self):
        # 文案大模型相关
        self.MODEL_API_KEY = os.getenv("MODEL_API_KEY")
        self.MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
        self.MODEL_NAME = os.getenv("MODEL_NAME")

        # 兼容 DeepSeek 单独配置
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or self.MODEL_API_KEY
        self.DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or self.MODEL_BASE_URL
        self.DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME") or self.MODEL_NAME

        # 读取极梦/火山引擎密钥
        self.JIMENG_AK = os.getenv("JIMENG_AK")
        self.JIMENG_SK = os.getenv("JIMENG_SK")


if __name__ == '__main__':
    conf = Config()
    print(conf.MODEL_BASE_URL)
