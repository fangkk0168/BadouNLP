#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

"""
作业题目：实现基于kmeans结果类内距离的排序。
"""


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


# 每个标题分词，然后用空格作为间隔
def load_sentence(path):
    # 创建一个空集合，主要是为了去重
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            # 使用jieba分词进行分词
            words = jieba.cut(sentence)
            sentences.add(" ".join(words))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        # 初始化一个词向量大小的维度
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算向量的欧式距离
def cal_distance(x, y):
    return np.linalg.norm(y - x)


if __name__ == "__main__":
    model = load_word2vec_model('model.w2v')  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    count = 0
    distance_label_dict = dict()
    for center_vector in np.array(kmeans.cluster_centers_):
        list = sentence_label_dict.get(count)
        drop_distance = 0
        for sentence in list:
            vectors = sentences_to_vectors(sentences, model)
            drop_distance += cal_distance(center_vector, vectors)
        distance_label_dict.setdefault(count, drop_distance / len(list))
        # distance_label_dict[count].append(drop_distance / len(list))
        count += 1

    print('所有的质心平均距离：', distance_label_dict)

    # 给质心平均距离，按照从大到小进行排序
    distance_label_dict_JX = sorted(distance_label_dict.items(), key=lambda x: x[1], reverse=True)

    # 循环打印分类结果查看
    for label, sentences in sentence_label_dict.items():
        print("cluster : %s，质心平均距离：%s" % (label, distance_label_dict.get(label)))
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
