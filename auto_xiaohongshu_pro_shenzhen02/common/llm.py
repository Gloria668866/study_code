from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from common.config import Config

conf = Config()


def build_deepseek_llm():
    if conf.DEEPSEEK_API_KEY and conf.DEEPSEEK_BASE_URL and conf.DEEPSEEK_MODEL_NAME:
        return ChatOpenAI(
            api_key=conf.DEEPSEEK_API_KEY,
            base_url=conf.DEEPSEEK_BASE_URL,
            model=conf.DEEPSEEK_MODEL_NAME,
        )
    return None


my_llm = build_deepseek_llm()


def get_llm():
    if my_llm is None:
        raise RuntimeError("请先配置 DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL、DEEPSEEK_MODEL_NAME")
    return my_llm


if __name__ == '__main__':
    print(get_llm()([HumanMessage(content="你好")]))
