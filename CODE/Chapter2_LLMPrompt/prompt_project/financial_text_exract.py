from langchain_openai import ChatOpenAI
import json
import re
from llm_output_parse_tool import parse_llm_json_output

API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# 步骤1： 定义金融实体的字段
SCHEMA = {
    "金融": ["日期", "股票名称", "开盘价", "收盘价", "成交量"]
}

# 定义 few-shot 示例，用于引导模型进行信息抽取
IE_EXAMPLES = {
    "金融": [
        {
            "content": "2023-01-10，股票古哥-D[EOOE]开盘价100美元，收盘价102美元，成交量520000。",
            "answers": {
                "日期": ["2023-01-10"],
                "股票名称": ["古哥-D[EOOE]"],
                "开盘价": ["100美元"],
                "收盘价": ["102美元"],
                "成交量": ["520000"]
            }
        }
    ]
}
print("=== 抽取字段 ===")


# 步骤 2.构造 Prompt 和推理逻辑
def extract_financial_info(sentences: list) -> list:
    """
    # 步骤2：从句子中抽取金融实体，输出 JSON 格式结果。
    """
    # 步骤 2.1：初始化 ChatOpenAI 模型
    llm = ChatOpenAI(
        model=MODEL,  # 指定 DeepSeek 模型
        api_key=API_KEY,  # API 密钥
        base_url=API_URL,  # API 基础 URL
        temperature=0.7,  # 控制生成随机性
        max_tokens=200  # 限制输出长度
    )

    # 步骤 2.2：构造 few-shot Prompt 前缀
    properties_str = ", ".join(SCHEMA["金融"])  # 将字段列表转为逗号分隔的字符串
    print(f"待抽取的字段：{properties_str}")
    prompt_prefix = (
        "你是一个金融信息抽取器，任务是从句子中提取‘金融’实体（日期、股票名称、开盘价、收盘价、成交量）。"
        "未提及的信息用 ['原文中未提及'] 表示。输出 JSON 格式。\n"
        "示例：\n"
        f"User: {IE_EXAMPLES['金融'][0]['content']}\n"
        f"提取上述句子中“金融”({properties_str})的实体，并按照 JSON 格式输出，未提及的信息用 ['原文中未提及'] 表示。\n"
        f"assistant: {json.dumps(IE_EXAMPLES['金融'][0]['answers'], ensure_ascii=False)}\n"
    )
    # print(prompt_prefix)

    # 步骤 2.3：遍历句子，执行信息抽取
    results = []
    for sentence in sentences:
        # 构造单句 Prompt
        prompt = (
            f"{prompt_prefix}"
            f"User: {sentence}\n"
            f"提取上述句子中“金融”({properties_str})的实体，并按照 JSON 格式输出，未提及的信息用 ['原文中未提及'] 表示。"
        )

        # print(prompt)
        # 步骤 2.4：调用模型并清理输出
        response = llm.invoke(prompt).content.strip()
        cleaned_response = parse_llm_json_output(response)
        results.append(cleaned_response)

    # 步骤 2.5：返回抽取结果
    return results

# 步骤 2.6：主程序，执行信息抽取并输出
if __name__ == "__main__":
    # 步骤 2.6.1：定义待抽取的句子
    sentences = [
        "2023-02-15，股票佰笃[BD]美股开盘价10美元，收盘价13美元，成交量460,000。",
        "2023-04-05，股票盘古(0021)开盘价23元，收盘价26美元，成交量310,000。"
    ]
    # 步骤 2.6.2：执行信息抽取
    results = extract_financial_info(sentences)

    # 步骤 2.3：打印抽取结果
    print("=== 信息抽取结果 ===")
    print(results)