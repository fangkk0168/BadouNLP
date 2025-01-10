"""
数据加载
"""

import codecs
import csv
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.index_to_label = {0: '差评', 1: '好评'}
        # {'差评': 0, '好评': 1}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        # 字典的长度放入配置文件中
        self.config['class_num'] = len(self.index_to_label)
        self.vocab = load_vocab(config['vocab_path'])
        # 词表长度放入配置文件中
        self.config['vocab_size'] = len(self.vocab)
        self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrain_model_path'])
        self.load_text()

    def load_text(self):
        self.data = []
        text_length = []
        with codecs.open(self.config['train_data_path'], encoding='UTF-8-sig') as f:
            # 这里读取的数据，都是json格式的。例 {'label': '1', 'review': '很快，好吃，味道足，量大'}
            for line in csv.DictReader(f, skipinitialspace=True):
                label = line['label']
                text = line['review']
                # 计算所有字符的平均长度
                text_length.append(len(text))
                if self.config['model_type'] == 'bert':
                    input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], padding='max_length',
                                                     truncation=True)
                else:
                    input_id = self.encode_sentence(text)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([int(label)])
                self.data.append([input_id, label_index])
        # 文本所有的长度放入Config中，后面计算文本平均长度
        self.config['text_length'] = text_length
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 加载词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf-8') as f:
        for index, value in enumerate(f):
            token = value.strip()
            token_dict[token] = index + 1
    return token_dict


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    # 模拟加载数据测试
    dg = DataGenerator(r'./text.csv', Config)
    print(dg[1])
