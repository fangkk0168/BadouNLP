"""
加载数据层
"""
import json
import jieba
import torch
import random
from collections import defaultdict
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        self.train_data_size = config['epoch_data_size']
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = 'train'
                    questions = line['questions']
                    label = line['target']
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 按照TripletLoss要求，选取a,p,n点。从不同的样本里面抽取
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 随机选择两个标签
        sample_lst = random.sample(standard_question_index, 2)
        label1 = sample_lst[0]
        label2 = sample_lst[1]
        # 如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        if len(self.knwb[label1]) < 2:
            return self.random_train_sample()
        else:
            # 同一个类别里面选择两个类似的点，a锚点，p与a统一类别的样本
            a, p = random.sample(self.knwb[label1], 2)
            # 随机从其他标签中选择一个负样本n点
            n = random.sample(self.knwb[label2], 1)
            return a, p, n[0]


# 加载词典方法
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
        return token_dict


# 加载规则库
def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    dg = DataGenerator(Config['train_data_path'], Config)
