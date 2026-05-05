# 导入 PyTorch 库
import torch
# 从 transformers 库导入所需的类
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 您原始代码的加载部分 ---
model_path = r"D:\models\Qwen2.5-3B-Instruct"
# 注意：为了仅计算参数量，我们无需加载分词器或将模型移动到 GPU
# tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16) # 使用 torch_dtype 加快加载速度，但对于计数无影响
# model.eval() # 对于参数计数，也无需切换到评估模式

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
