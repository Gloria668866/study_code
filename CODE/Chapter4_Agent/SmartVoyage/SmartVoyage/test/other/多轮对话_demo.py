import logging
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from config import Config

def initialize_llm():
    """初始化 LLM"""
    conf = Config()
    try:
        return ChatOpenAI(
            model=conf.model_name,
            api_key=conf.api_key,
            base_url=conf.api_url,
            temperature=0.7
        )
    except Exception as e:
        logger.error(f"LLM 初始化失败: {str(e)}")
        raise


def initialize_conversation():
    """初始化对话链，带有记忆功能"""
    llm = initialize_llm()
    memory = ConversationBufferMemory()  # 创建对话记忆
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False  # 设置为 True 可查看详细日志
    )
    return conversation


def chat_with_bot():
    """运行多轮对话"""
    print("欢迎使用聊天机器人！输入 '退出' 结束对话。")
    conversation = initialize_conversation()

    while True:
        user_input = input("\n您: ")
        if user_input.strip().lower() == "退出":
            print("对话已结束！")
            break

        try:
            response = conversation.predict(input=user_input)
            print(f"机器人: {response}")
        except Exception as e:
            print(f"发生错误: {str(e)}")
            logger.error(f"对话处理失败: {str(e)}")


if __name__ == "__main__":
    chat_with_bot()