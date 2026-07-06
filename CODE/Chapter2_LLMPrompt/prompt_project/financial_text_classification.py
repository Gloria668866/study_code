from langchain_openai import ChatOpenAI
import json

# DeepSeek API 配置
API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# 初始化 ChatOpenAI
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    temperature=0.7,
    max_tokens=150
)

# Few-shot 示例
CLASS_EXAMPLES = {
    "新闻报道": "今日，股市经历震荡，投资者关注美联储政策调整。",
    "财务报告": "公司年度财务报告显示盈利增长，资产负债表强劲。",
    "公司公告": "本公司完成并购交易，收购人工智能领域领先公司。",
    "分析师报告": "行业分析指出，科技公司创新是未来增长关键。"
}

# 待分类句子
sentences = [
    "今日，央行发布公告宣布降低利率，以刺激经济增长。",
    "ABC公司今日发布公告称，已成功完成对XYZ公司股权的收购交易。",
    "公司资产负债表显示，公司偿债能力强劲，现金流充足。",
    "最新的分析报告指出，可再生能源行业预计将在未来几年经历持续增长。"
]


if __name__ == '__main__':
    # 构造 Prompt 和推理
    results = []
    class_list = list(CLASS_EXAMPLES.keys()) #["新闻报道","财务报告","公司公告","分析师报告"]
    # 初始化 Few-shot Prompt
    prompt_prefix = (
        f"你是一个金融文本分类器，任务是将输入句子分类到以下类别之一：{class_list}。逐一分析句子，返回 JSON 格式结果："
        "示例：\n"
        f"User: “{CLASS_EXAMPLES['新闻报道']}” 是 {class_list} 里的什么类别？ {{results：新闻报道}}"
        f"User: “{CLASS_EXAMPLES['财务报告']}” 是 {class_list} 里的什么类别？{{results：财务报告}}"
        f"""返回格式必须是json"""
    )


    # print(prompt_prefix)

    for sentence in sentences:
        # 构造单句 Prompt
        prompt = f"{prompt_prefix}User: “{sentence}” 是 {class_list} 里的什么类别？"
        # print(prompt)
        response = llm.invoke(prompt).content.strip()
        results.append(response)

    # 输出结果
    output = {"results": results}
    print("=== 文本分类结果 ===")
    print(output)


