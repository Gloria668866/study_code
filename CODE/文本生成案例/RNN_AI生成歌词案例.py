




import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time




def build_vocab():
    #读取文件
    #1.定义变量 记录去重之后的词，每行文本分词结果
    unique_words = []
    all_words = []
    file_path = 'data/jaychou_lyrics.txt'
    for line in open(file=file_path, mode='r', encoding='utf-8'):
        # 使用jieba分词,分割结果是一个列表
        words = jieba.lcut(line)
        # print('words->', words)
        # 所有的分词结果存储到all_words，其中包含重复的词组
        # [[],[],[]]
        all_words.append(words)
        # 遍历分词结果，去重后存储到unique_words
        # words->['想要', '有', '直升机', '\n']
        for word in words:
            # word词不再词汇表中, 就是一个新词
            if word not in unique_words:
                unique_words.append(word)
        # 统计去重语料中词的数量
    word_count=len(unique_words)
    #构建词表
    #例如：{'想要'：0，‘有’:1}
    word_to_index = {}
    for idx, word in enumerate(unique_words):
         word_to_index[word] = idx
    #歌词文本用索引表示
    corpus_index = []
    for words in all_words:
        #定义变量，记录：词索引列表
        tmp=[]
        #获取每一行的词，并获取相应的索引
        for word in words:
             tmp.append(word_to_index[word])
         # 在每行词之间添加空格隔开
        tmp.append(word_to_index[' '])
         #获取文档中每个词的索引，添加到corpus_index中
        corpus_index.extend(tmp)
# 获取当中每个词对应的索引
    return unique_words, word_to_index, word_count, corpus_index

# 构建数据集对象型
# 创建类继承 torch.utils.data.Dataset基类
class LyricsDataset(torch.utils.data.Dataset):
    # 定义构造方法
    # corpus_idx: 歌词文本用词表索引表示
    # num_chars: 每句话的词数量
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        # 统计歌词文本中有多少个词, 不去重
        self.word_count = len(corpus_idx)
        # 歌词文本能生成多少个句子
        self.number = self.word_count // self.num_chars

    # 重写__len__魔法方法, len()->输出返回值
    def __len__(self):
        return self.number

    # 重写__getitem__魔法方法 obj[idx]执行此方法 遍历数据加载器执行此方法
    def __getitem__(self, idx):
        # 设置起始下标值 start, 不能超过word_count-num_chars-1
        # -1: y的值要x基础上后移一位
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        # 获取x
        x = self.corpus_idx[start: end]
        y = self.corpus_idx[start + 1: end + 1]
        return torch.tensor(x), torch.tensor(y)




# 模型构建
class TextGenerator(nn.Module):
    def __init__(self, unique_word_count):
        super(TextGenerator, self).__init__()
        # 初始化词嵌入层: 语料中词的数量, 词向量的维度为128
        self.ebd = nn.Embedding(unique_word_count, 128)
        # 循环网络层: 词向量维度128, 隐藏向量维度256, 网络层数1
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层: 特征向量维度256与隐藏向量维度相同, 词表中词的个数
        self.out = nn.Linear(256, unique_word_count)

    def forward(self, inputs, hidden):
        # 输出维度: (batch, seq_len, 词向量维度128)
        # batch：句子数量
        # seq_len： 句子长度， 每个句子由多少个词 词数量
        embed = self.ebd(inputs)
        # rnn层x的表示形式为(seq_len, batch, 词向量维度128)
        # output的表示形式与输入x类似，为(seq_len, batch, 词向量维度256)
        # 前后的hidden形状要一样, 所以DataLoader加载器的batch数要能被整数
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)
        # 全连接层输入二维数据， 词数量*词维度
        # 输入维度: (seq_len*batch, 词向量维度256)
        # 输出维度: (seq_len*batch, 语料中词的数量)
        # output: 每个词的分值分布，后续结合softmax输出概率分布
        # output.shape[-1]: 词向量维度表示
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 网络输出结果
        return output, hidden

    def init_hidden(self, bs):
        # 隐藏层的初始化:[网络层数, batch, 隐藏层向量维度]
        return torch.zeros(1, bs, 256)





def train():
    # 构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 数据集 LyricsDataset对象，并实现了 __getitem__ 方法
    lyrics = LyricsDataset(corpus_idx=corpus_idx, num_chars=32)
    # 查看句子数量
    # print(lyrics.number)
    # 初始化模型
    model = TextGenerator(unique_word_count)
    # 数据加载器 DataLoader对象，并将lyrics dataset对象传递给它
    lyrics_dataloader = DataLoader(lyrics, shuffle=True, batch_size=5)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 训练轮数
    epoch = 50
    for epoch_idx in range(epoch):
        # 训练时间
        start = time.time()
        iter_num = 0  # 迭代次数
        # 训练损失
        total_loss = 0.0
        # 遍历数据集 DataLoader 会在后台调用 dataset.__getitem__(index) 来获取每个样本的数据和标签，并将它们组合成一个 batch
        for x, y in lyrics_dataloader:
            #print('y.shape->', y.shape)
            # 隐藏状态的初始化
            hidden = model.init_hidden(bs=5)
            # 模型计算
            output, hidden = model(x, hidden)
            #print('output.shape->', output.shape)
            # 计算损失
            # y形状为(batch, seq_len), 需要转换成一维向量->160个词的下标索引
            # output形状为(seq_len, batch, 词向量维度)
            # 需要先将y进行维度交换(和output保持一致)再改变形状
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1  # 迭代次数加1
            total_loss += loss.item()
        # 打印训练信息
        print('epoch %3s loss: %.5f time %.2f' % (epoch_idx + 1, total_loss / iter_num, time.time() - start))
    # 模型存储
    torch.save(model.state_dict(), 'model/lyrics_model_%d.pth' % epoch)

def predict(start_word, sentence_length):
    # 构建词典
    unique_words, word_to_index, unique_word_count, _ = build_vocab()
    # 构建模型
    model = TextGenerator(unique_word_count)
    # 加载参数
    model.load_state_dict(torch.load('model/lyrics_model_10.pth'))
    # 隐藏状态
    hidden = model.init_hidden(bs=1)
    # 将起始词转换为索引
    word_idx = word_to_index[start_word]
    # 产生的词的索引存放位置
    generate_sentence = [word_idx]
    # 遍历到句子长度，获取每一个词
    for _ in range(sentence_length):
        # 模型预测
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        # 获取预测结果
        word_idx = torch.argmax(output)
        generate_sentence.append(word_idx)
    # 根据产生的索引获取对应的词，并进行打印
    for idx in generate_sentence:
        print(unique_words[idx], end='')



if __name__ == '__main__':
    # 调用预测函数
    predict('星星', 50)

























