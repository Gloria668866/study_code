try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None

from common.config import Config

conf = Config()

if ChatOpenAI is not None:
    my_llm = ChatOpenAI(
        api_key=conf.DEEPSEEK_API_KEY,
        base_url=conf.DEEPSEEK_BASE_URL,
        model=conf.DEEPSEEK_MODEL_NAME,
    )
else:
    my_llm = None


if __name__ == '__main__':
    if my_llm is None:
        print("langchain_openai is not installed")
    else:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="用一句话介绍一下你自己")]
        response = my_llm.invoke(messages)
        print(response.content)
