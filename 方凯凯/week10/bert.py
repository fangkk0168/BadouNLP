# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
from transformers import BertTokenizer
from transformers import BertModel

"""
基于pytorch的Bert语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"D:\code_repository\ai-study\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        # print('输入x的形状：', x.shape)
        if mask is None:
            sequence_output, pool_output = self.bert(x)
        else:
            if torch.cuda.is_available():
                x = x.cuda()
                mask = mask.cuda()
            sequence_output, pool_output = self.bert(input_ids=x, attention_mask=mask)
        y_pred = self.classify(sequence_output)  # output shape:(batch_size, seq_length, vocab_size)
        if y is not None:
            """
            y_pred：64 * 10 * 3961。可以理解为64个12*3961的矩阵
                y_pred.shape[-1]：3961
                y_pred.view(-1, 3961)：640 * 3961。把64个10*3961矩阵都放在一起，可以想象成摞起来放在一起，也就是 640 * 3961的形状。
            y：64 * 10。可以理解为64*10的矩阵，也就是64个长度为10的向量，如下所示：
                [[ 1,2,3,4,5,6,7,8,9,10],
                 [ 1,2,3,4,5,6,7,8,9,10],
                 ...中间总共64个
                                        ]]
                y.view(-1)：640。view(-1)表示将Tensor转为一维Tensor。
                [ 1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10...] 总共640个元素
            """
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = [vocab.get(word, vocab['<UNK>']) for word in window]
    y = [vocab.get(word, vocab['<UNK>']) for word in target]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            """
            model(x)：得到的形状为[1,10,3961]
            model(x)[0]：得到的形状为[10,3961]
            model(x)[0][-1]：得到词表大小的概率分布 [第1个,...0.2,0.18,...,第3961个]
            """
            y = model(x)[0][-1]
            # 采样策略，获取到词表上某个字的索引值
            index = sampling_strategy(y)
            # 根据索引值获取具体的字，不断重复循环，直到句子换行后不再预测
            pred_char = reverse_vocab[index]
    return openings


# 采样策略
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        # 90%的概率，采用greedy
        strategy = "greedy"
    else:
        # 10%的概率，采用sampling
        strategy = "sampling"

    if strategy == "greedy":
        # 比较简单，直接获取概率值最高的
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        """
        prob_distributions：词表大小的概率分布 [第1个,...0.2,0.18,...,第3961个]
        len(prob_distribution)：3961
        range(3961)：range(0,3961)
        list(0,3961)：[0,1,2,...3961]
        np.random.choice：是从这个list集合中随机选择一个索引值，作为返回
        p=prob_distribution：概率值相加=1(因为是过了softmax函数，概率值相加是1)，按照概率值有权重地选择元素
        """
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    tokenizer = BertTokenizer.from_pretrained(r"D:\code_repository\ai-study\bert-base-chinese")
    model = build_model(vocab, char_dim)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            batch_size, seq_length = x.size()
            mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            loss = model(x, y, mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("corpus.txt", False)
