"""
模型层：使用
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.lstm(x)
        # 使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss(margin=config['margin'])

    def forward(self, sentence1, sentence2=None, sentence3=None):
        # 同时传入三个句子
        if sentence2 is not None and sentence3 is not None:
            anchor = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
            positive = self.sentence_encoder(sentence2)
            negative = self.sentence_encoder(sentence3)
            return self.loss(anchor, positive, negative)
        # 单独传入一个句子时，认为正在使用向量化能力
        if sentence1 is not None:
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    if config['model_type'] == "bert":
        learning_rate = config["bert_learning_rate"]
    else:
        learning_rate = config["other_learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    a = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    p = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    n = torch.LongTensor([[8, 9, 6, 7], [6, 7, 8, 9]])

    y_pred = model(a, p, n)
    print('TripletLoss预测值：', y_pred)
