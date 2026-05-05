import torch

file_name = "malicious_model.bin"

print(f"\n准备加载文件 '{file_name}'...")
print("如果这个文件是恶意的，下一行代码将触发隐藏的命令。")

try:
    # 模拟加载一个来路不明的模型文件
    # 这一步会触发 MaliciousCode 类的 __reduce__ 方法
    loaded_object = torch.load(file_name,weights_only=False)
    print("文件加载完成。") # 如果是真正的病毒的话，那么服务器就已经崩溃了
except Exception as e:
    print(f"加载过程中出现错误: {e}")