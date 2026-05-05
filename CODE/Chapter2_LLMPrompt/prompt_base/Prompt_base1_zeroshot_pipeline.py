from transformers import pipeline

model_path = r"D:\models\Qwen2.5-3B-Instruct"
generator = pipeline("text-generation", model=model_path, device=0)

prompt = """
你是传智教育的客服助手。课程套餐包括：
- AI 大模型开发工程师（3000 贝，12 周，程序员）
- AI 大模型数据分析工程师（2500 贝，10 周，数据分析师）
- AI 大模型运维工程师（2000 贝，8 周，运维工程师）
- AI 大模型 Java 开发工程师（3500 贝，15 周，Java 程序员）
根据用户输入，推荐合适的套餐，输出 JSON：{"推荐套餐": "名称", "理由": "说明"}。

用户输入：预算 2000 贝以内。
"""

# 调用生成器（generator）来根据给定的提示（prompt）生成文本
# 参数说明：
# - prompt: 输入的文本提示，用于指导生成器生成相应的文本
# - truncation=True: 启用截断，确保输入不会超过模型的最大长度限制
# - num_return_sequences=1: 指定生成器只返回一个生成结果
# - pad_token_id=generator.tokenizer.pad_token_id: 设置填充标记的ID，用于对输入序列进行填充（通常用于保持输入长度一致）
result = generator(prompt, truncation=True, num_return_sequences=1, pad_token_id=generator.tokenizer.pad_token_id)

# 打印生成结果中的第一个（也是唯一一个）生成的文本
# result 是一个列表，包含生成的序列；result[0] 访问第一个序列
# ["generated_text"] 提取该序列中的生成文本内容
print("生成结果：")
print(result[0]["generated_text"])



"""
prompt从结果中输出，题可能与以下原因有关：
模型理解不足：Qwen2.5-3B-Instruct 可能在理解预算约束或严格遵循 JSON 格式要求时表现不佳。
Prompt 设计：prompt 虽然清晰，但可能需要更明确的指令（比如“仅输出 JSON，不包含其他文本”）。
生成参数：no_repeat_ngram_size=2 可能限制了模型的表达，导致生成内容不够准确。
"""