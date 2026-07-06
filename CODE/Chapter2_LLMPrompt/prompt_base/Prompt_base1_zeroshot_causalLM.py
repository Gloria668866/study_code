from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 定义模型路径，指向本地存储的 Qwen2.5-3B-Instruct 模型
model_path = r"D:\models\Qwen2.5-3B-Instruct"

# 加载预训练的分词器（tokenizer），用于将文本转换为模型可处理的 token
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载预训练的因果语言模型（CausalLM），并配置：
# - torch_dtype=torch.float16：使用半精度浮点数以节省内存和加速推理
# - device_map="auto"：自动将模型分配到可用设备（GPU 或 CPU）
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# 将模型设置为评估模式（禁用 dropout 等训练时的随机性）
model.eval()

# 定义 prompt，包含任务描述（客服助手、课程套餐信息）和用户输入
prompt = """
你是传智教育的客服助手。课程套餐包括：
- AI 大模型开发工程师（3000 贝，12 周，程序员）
- AI 大模型数据分析工程师（2500 贝，10 周，数据分析师）
- AI 大模型运维工程师（2000 贝，8 周，运维工程师）
- AI 大模型 Java 开发工程师（3500 贝，15 周，Java 程序员）
根据用户输入，推荐合适的套餐，输出 JSON：{"推荐套餐": "名称", "理由": "说明"}。

用户输入：预算 2000 贝以内。
"""

# 使用分词器将 prompt 转换为模型输入格式（张量）
# - return_tensors="pt"：返回 PyTorch 张量
# - .to("cuda")：将输入张量移动到 GPU 上
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 使用模型生成文本
# - inputs["input_ids"]：输入的 token ID 序列
# - attention_mask=inputs["attention_mask"]：注意力掩码，标记有效 token
# - max_length=1000：生成文本的最大长度（包括 prompt 和生成内容）
# - num_return_sequences=1：生成一个序列
# - pad_token_id=tokenizer.pad_token_id：设置填充 token 的 ID
outputs = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=1000,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id
)

# 解码生成的 token 序列为文本
# - outputs[0]：取第一个（也是唯一）生成的序列
# - skip_special_tokens=True：跳过特殊 token（如结束标记）
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印解码后的结果
print(result)


#
# # 计算 prompt 的长度（以字符为单位）
# prompt_length = len(prompt)
#
# # 解码时只取生成的部分
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# generated_text = result[prompt_length:].strip()  # 去掉 prompt 部分并移除多余空格
# print(generated_text)