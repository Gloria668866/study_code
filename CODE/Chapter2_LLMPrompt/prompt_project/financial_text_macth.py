from rich import print
from langchain_openai import ChatOpenAI

# 步骤 1.1：为 DeepSeek API 配置添加注释
API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# 步骤 1.2：为 examples 字典添加注释=》few-shot示例
# 结构为 Dict[str, List[Tuple[str, str]]]，键 '是' 表示相似，'不是' 表示不相似
examples = {
    '是': [('公司ABC发布了季度财报，显示盈利增长。', '财报披露，公司ABC利润上升。'),],
    '不是': [('黄金价格下跌，投资者抛售。', '外汇市场交易额创下新高。'),('央行降息，刺激经济增长。', '新能源技术的创新。')]
}

# 步骤 1.1：定义辅助函数，生成提示字符串
def build_prompt(sentence1: str, sentence2: str) -> str:
    """
    生成语义相似性判断的提示字符串。
    """
    return f'句子一:{sentence1}\n句子二:{sentence2}\n上面两句话是相似的语义吗？'

# 步骤 1.2：定义任务指令和初始回答常量
TASK_INSTRUCTION = ('现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答“是”或“不是”，不要做多余的回答。')
INITIAL_RESPONSE = '好的，我将只回答“是”或“不是”。'
def init_prompts():
    """
    # 步骤 2.1： 初始化前置 prompt，用于模型进行 in-context learning。
    """
    # 步骤 2.2.1：初始化历史对话，包含任务指令和初始回答
    pre_history = [(TASK_INSTRUCTION, INITIAL_RESPONSE)]
    # print(">>> 前置提示：", pre_history)
    # 步骤 2.2.2：初始化临时列表，存储 few-shot 示例
    few_shot_examples = []
    # 步骤 2.2.3：遍历 examples 生成 few-shot 示例
    for key, sentence_pairs in examples.items():
        # 遍历每个键（'是' 或 '不是'）对应的句子对列表
        for sentence1, sentence2 in sentence_pairs:
            # 生成提示字符串
            prompt = build_prompt(sentence1, sentence2)
            # 将提示和对应的回答（key）添加到临时列表
            few_shot_examples.append((prompt, key))
    # print(">>> few-shot 示例：", few_shot_examples)
    # 步骤 2.2.4：将 few-shot 示例追加到 pre_history
    pre_history.extend(few_shot_examples)
    print(">>> 含few-shot的历史对话：", pre_history)
    # 步骤 2.2.5：返回包含历史对话的字典
    return {"pre_history": pre_history}
def inference(sentence_pairs: list, custom_settings: dict, model: ChatOpenAI):
    """
    # 步骤 3.1：推理函数，判断输入句子对的语义是否相似。
    """
    # 步骤 3.2.1：遍历每对输入句子
    for sentence_pair in sentence_pairs:
        sentence1, sentence2 = sentence_pair
        # 步骤 3.2.2：构造推理提示，询问两句话的语义是否相似
        sentence_with_prompt = f'句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗？'
        print("=====================================")
        print(sentence_with_prompt)

        # 步骤 3.2.3：构造消息列表
        messages = []
        # 添加历史对话（few-shot 示例）
        for user_msg, assistant_msg in custom_settings['pre_history']:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        # 添加当前推理提示
        messages.append({"role": "user", "content": sentence_with_prompt})

        # print(messages)

        # 步骤 3.2.4：调用 ChatOpenAI 模型进行推理
        try:
            response = model.invoke(messages)  # 发送请求并获取响应
            answer = response.content.strip()  # 提取模型回答并去除多余空格
        except Exception as e:
            answer = f"推理失败: {str(e)}"  # 捕获异常并返回错误信息

        # 步骤 3.2.5：打印句子对和推理结果
        print(f'>>> {sentence_pair}')
        print(f'>>> {answer}')

if __name__ == '__main__':
    # 步骤 1.3.1：初始化 ChatOpenAI 模型，配置 DeepSeek API
    model = ChatOpenAI(
        api_key=API_KEY,  # API 密钥，用于认证
        base_url=API_URL,  # API 基础 URL，指定请求端点
        model_name=MODEL,  # 模型名称，指定 DeepSeek 模型
        max_tokens=50,  # 限制最大输出 token 数，确保简洁输出
        temperature=0.7  # 控制生成随机性，0.7 为适中值
    )

    # 步骤 1.3.2：定义待推理的句子对
    sentence_pairs = [
        ('股票市场今日大涨，投资者乐观。', '持续上涨的市场让投资者感到满意。'),
        ('油价大幅下跌，能源公司面临挑战。', '未来智能城市的建设趋势愈发明显。'),
        ('利率上升，影响房地产市场。', '高利率对房地产有一定冲击。'),
    ]
    # 步骤 1.3.3：初始化 few-shot 示例并执行推理
    print('>>> few-shot 示例：')
    print(init_prompts())
    custom_settings = init_prompts()  # 调用 init_prompts 生成 few-shot 示例
    # print(">>> 自定义设置：", custom_settings)

    inference(sentence_pairs, custom_settings, model)  # 执行推理并打印结果