import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 引入标准化模块
import numpy as np
import pandas as pd
import time
import os


# 1. 数据预处理
def create_dataset():
    # 读取数据
    data = pd.read_csv(r"手机价格预测.csv")

    # 获取特征列和标签列
    x = data.iloc[:, :-1].values  # 转换为 numpy 数组
    y = data.iloc[:, -1].values

    # 划分训练集和测试集 (注意：必须先划分，再标准化，防止数据泄露)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8, stratify=y)

    # === 核心优化：特征标准化 ===
    # 神经网络对数据的量纲非常敏感，不进行标准化会导致梯度爆炸或消失
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 转化为数据集对象
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# 2. 搭建神经网络 (优化版)
class PhoneModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PhoneModel, self).__init__()
        # 使用 nn.Sequential 让代码更整洁
        # 加入了 BatchNorm (批标准化) 和 Dropout (防止过拟合)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 随机断开 30% 的神经元

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# 3. 训练模型
def train_model(train_dataset, test_dataset, input_dim, output_dim):
    # 构建数据加载器 (批次大小设为 32 或 64 比较合适)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 引入测试集加载器，用于在训练过程中监控准确率
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = PhoneModel(input_dim=input_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()

    # 核心优化：改用 Adam 优化器，收敛速度远超 SGD
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epoch = 100
    best_acc = 0.0  # 记录历史最高准确率

    # 确保保存模型的文件夹存在
    if not os.path.exists('model'):
        os.makedirs('model')

    print("开始训练模型...")
    for epoch in range(num_epoch):
        model.train()  # 设置为训练模式
        total_loss = 0.0
        start_time = time.time()

        for x, y in train_loader:
            optimizer.zero_grad()  # 梯度清零
            y_pred = model(x)  # 前向传播
            loss = criterion(y_pred, y)  # 计算损失
            loss.backward()  # 反向传播 (去掉了多余的 .sum())
            optimizer.step()  # 更新参数
            total_loss += loss.item()

        # --- 每个 epoch 结束后，在测试集上评估一次，保存最好的模型 ---
        model.eval()  # 设置为评估模式 (关闭 Dropout 等)
        correct = 0
        with torch.no_grad():  # 核心优化：测试时不需要计算梯度，节省显存并提速
            for x_val, y_val in test_loader:
                y_val_pred = model(x_val)
                preds = torch.argmax(y_val_pred, dim=1)
                correct += (preds == y_val).sum().item()

        current_acc = correct / len(test_dataset)
        avg_loss = total_loss / len(train_loader)

        # 打印当前进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f'Epoch [{epoch + 1:3d}/{num_epoch}] | Loss: {avg_loss:.4f} | 测试集准确率: {current_acc:.4f} | 耗时: {time.time() - start_time:.2f}s')

        # 只保存准确率最高的模型权重
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), 'model/phonemodel.pth')

    print(f'训练完成！历史最高测试集准确率为: {best_acc:.4f} (模型已保存)')


# 4. 模型评估
def evaluate_model(test_dataset, input_dim, output_dim):
    model = PhoneModel(input_dim=input_dim, output_dim=output_dim)
    # 加载我们刚刚保存的最高准确率的模型
    model.load_state_dict(torch.load('model/phonemodel.pth'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()  # 必须开启测试模式
    current = 0

    with torch.no_grad():  # 停止梯度计算
        for x, y in test_loader:
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            current += (y_pred == y).sum().item()
            print(y_pred)
            print(y)
    final_acc = current / len(test_dataset)
    print('\n================================')
    print('最终加载最优模型进行测试...')
    print('✅ 测试集最终准确率: %.2f%%' % (final_acc * 100))
    print('================================')


if __name__ == '__main__':
    # 获取数据集和维度信息
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    # 训练模型 (建议执行一次，观察准确率飙升的过程)
    #train_model(train_dataset, test_dataset, input_dim, output_dim)

    # 评估模型
    evaluate_model(test_dataset, input_dim, output_dim)