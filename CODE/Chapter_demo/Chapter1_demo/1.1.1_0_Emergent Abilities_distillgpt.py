from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
# 禁用所有日志
logging.getLogger("transformers").setLevel(logging.CRITICAL)
# 模型名称 distilgpt2少量参数
# model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir=r"D:\models")
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir=r"D:\models")


# 模型名称 参数多一些 Qwen2.5-0.5B
model = AutoModelForCausalLM.from_pretrained(r"D:\models\Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained(r"D:\models\Qwen2.5-0.5B")

# 使用模型进行简单的推理
# prompt = "The capital of china is"
prompt = """
        请将下面的中文翻译成英文
        
        中文: 我喜欢吃苹果
        English:
        """
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
print("推理结果:", tokenizer.decode(outputs[0], skip_special_tokens=True))




# --- 参数量计算部分 ---

# 初始化总参数计数器为 0
total_params = 0

# model.parameters() 会返回一个包含模型所有可训练参数（权重和偏置）的迭代器
# 这里的参数主要是神经网络中各个层的权重（weights）和偏置（biases）
for param in model.parameters():
    # param 是一个 PyTorch 张量 (torch.Tensor)
    # p.numel() 方法会返回该张量中元素的总数量
    # 通过累加每个张量的元素数量，我们就能得到模型的总参数量
    total_params += param.numel()

# --- 格式化输出结果 ---

# 为了让人更容易阅读，我们将总参数量转换为以“亿”为单位的浮点数
# 1 亿 = 100,000,000  tips：1000000000与1_000_000_000等价
total_params_in_billion = total_params / 1_000_000_000

# 打印精确的总参数量
print(f"模型的精确总参数量为: {total_params}")

# 打印以“亿”为单位的参数量，保留两位小数
print(f"模型参数量（约）: {total_params_in_billion:.2f} B")



