from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
# 禁用所有日志
logging.getLogger("transformers").setLevel(logging.CRITICAL)
# 模型名称
model_id = "distilgpt2" #353M

print(f"正在加载 PyTorch 格式的模型 ({model_id})...")
print("通过设置 use_safetensors=False 来确保加载 .bin 文件。")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    cache_dir=r"D:\models",
    use_safetensors=False  # <-- 核心在这里：添加此参数
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=r"D:\models"
)

print("PyTorch 模型 (.bin) 加载完成。")

# 使用模型进行简单的推理
prompt = "The capital of china is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
print("推理结果:", tokenizer.decode(outputs[0], skip_special_tokens=True))