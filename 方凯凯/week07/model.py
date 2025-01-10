"""
模型层
"""
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config
        vocab_size = config['vocab_size'] + 1
        hidden_size = config['hidden_size']
        class_num = config['class_num']
        model_type = config['model_type']
        num_layers = config["num_layers"]
        self.use_bert = False

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == 'fast_text':
            self.encoder = lambda x: x
        elif model_type == 'rnn':
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == 'bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失

    def forward(self, x, target=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]

        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)

        # 也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]
        predict = self.classify(x)  # input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


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

    Config["class_num"] = 2
    Config["vocab_size"] = 20
    Config["model_type"] = "dnn"

    model = TorchModel(config=Config)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    pred = model(x)
    print(pred)
