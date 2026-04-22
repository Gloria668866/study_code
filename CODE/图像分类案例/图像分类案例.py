import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8



#创建数据集
def get_dataset():
    train_dataset = CIFAR10(root="data", train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root="data", train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset



#搭建神经网络
class moudel(nn.Module):
    def __init__(self):
        # 继承父类
        super().__init__()
        #第一个卷积层
        self.conv1=nn.Conv2d(3,6,3,1)
        #第一个池化层
        self.pool=nn.MaxPool2d(2,2)
        #第二个卷积层
        self.conv2=nn.Conv2d(6,16,3,1)
        #第二个池化层
        self.pool2=nn.MaxPool2d(2,2)
        #第一个全连接层
        self.lener1=nn.Linear(16*6*6,120)
        #第二个全连接层
        self.lener2=nn.Linear(120,84)
        #第三个全连接层 也是输出层
        self.output=nn.Linear(84,10)
    #定义前向传播
    def forward(self,x):
        # 第1层 卷积+激活+池化 计算
        x=self.pool(torch.relu(self.conv1(x)))
        # 第2层 卷积+激活+池化 计算
        x=self.pool2(torch.relu(self.conv2(x)))
        # 第1层隐藏层  只能接收二维数据集
        # 四维数据集转换成二维数据集
        # x.shape[0]: 每批样本数, 最后一批可能不够8个, 所以不是写死8
        x=x.reshape(shape=(x.shape[0],-1))#改变维度 因为全连接层只能处理二维数据

        x=torch.relu(self.lener1(x))

        x=torch.relu(self.lener2(x))

        return self.output(x)

#模型训练
def train_moudle(train_dataset):
    #   创建训练数据加载器对象
    dataloder=DataLoader(train_dataset,batch_size=BATCH_SIZE)

    module=moudel()

    # 损失函数
    certerion=nn.CrossEntropyLoss()

    # 反向传播 梯度清零＋误差计算 ＋ 参数更新
    optimizer=optim.Adam(module.parameters(),lr=0.001)
    epochos=20
    #开始训练
    for epoch in range(epochos):
        # 定义总损失变量
        total_loss = 0.0
        # 定义预测正确样本个数变量
        total_correct = 0
        # 定义总样本数据变量
        total_samples = 0
        # 定义开始时间变量
        start = time.time()
        for x,y in dataloder:
            #切换成为训练模式
            module.train()
            # 梯度清零
            optimizer.zero_grad()
            # 预测值
            y_pred=module(x)
            # 损失函数
            loss=certerion(y_pred,y)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            total_loss += loss.item()
            total_correct += (y_pred.argmax(-1) == y).sum().item()
            total_samples += y.size(0)
            # 打印每个 epoch 的结果
        end = time.time()
        print(f'Epoch {epoch + 1}/{epochos}, Loss: {total_loss / len(dataloder):.4f}, '
              f'Accuracy: {total_correct / total_samples * 100:.2f}%, '
              f'Time: {end - start:.2f}s')
    torch.save(obj=module.state_dict(), f='model/imagemodel.pth')

#模型评估
# 模型评估
def evaluate_model(test_dataset):
    # 创建测试集数据加载器
    dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 创建模型对象, 加载训练模型参数
    model = moudel()
    model.load_state_dict(torch.load('model/imagemodel.pth'))
    # 定义统计预测正确样本个数变量 总样本数据变量
    total_correct = 0
    total_samples = 0
    # 遍历数据加载器
    for x, y in dataloader:
        # 切换推理模型
        model.eval()
        # 模型预测
        output = model(x)
        # 将预测分值转成类别
        # 因为训练的时候用了CrossEntropyLoss 所以搭建神经网络时候没有搭建softmax 这个函数
        #所以要用argmax来模拟softmax -1表示行索引
        y_pred = torch.argmax(output, dim=-1)
        print('y_pred->', y_pred)
        # 统计预测正确的样本个数
        total_correct += (y_pred == y).sum()
        # 统计总样本数
        total_samples += len(y)

    # 打印精度
    print('Acc: %.2f' % (total_correct / total_samples))


if __name__ == '__main__':
    # 创建数据集
    train_dataset, test_dataset = get_dataset()
    # 训练模型
    #trained_model = train_moudle(train_dataset)

    # 评估模型
    evaluate_model(test_dataset)