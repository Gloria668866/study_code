# 导入 PyTorch 库
import torch
# 从 transformers 库导入所需的类
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 您原始代码的加载部分 ---
model_path = r"D:\models\Qwen2.5-3B-Instruct"
# 注意：为了仅计算参数量，我们无需加载分词器或将模型移动到 GPU
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path) # 使用 torch_dtype 加快加载速度，但对于计数无影响 ,默认32浮点数，即字节为4字节，显存占用为4*3B=12B 约为多少G:12B/1024/1024/1024=0.011G
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16) # 使用 torch_dtype 加快加载速度，但对于计数无影响
# model.eval() # 对于参数计数，也无需切换到评估模式
print(model.dtype)