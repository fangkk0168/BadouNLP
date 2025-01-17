"""
主程序层
"""

# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
作业：使用tripletloss完成文本匹配任务训练
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = SiameseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    start_time = datetime.datetime.now()
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        # logger.info("epoch %d begin" % epoch)
        print(f"第 {epoch} 轮 begin")
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            a, p, n = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(a, p, n)
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                print("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        print("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    end_time = datetime.datetime.now()
    time = (end_time - start_time).seconds
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return acc, evaluator, time


if __name__ == "__main__":
    for model in ["fast_text"]:
        Config["model_type"] = model
        acc, evaluator, time = main(Config)
        if model == 'bert':
            learning_rate = Config["bert_learning_rate"]
        else:
            learning_rate = Config["other_learning_rate"]
        print('| 模型名称：', model,
              '| 学习率：', format(learning_rate, 'f'),
              '| Hidden_size：', Config["hidden_size"],
              '| Batch_size：', Config["batch_size"],
              '| 正确样本数：', evaluator.stats_dict['correct'],
              '| 错误样本数：', evaluator.stats_dict['wrong'],
              '| 总耗时(秒)：', str(time) + '秒',
              '| 文本最大长度：', Config["max_length"],
              '| 最后一轮准确率：', acc,
              )
        # 组装表格数据
        output_result_list = []
        model_dict = {}
        model_dict.setdefault('模型名称', model)
        model_dict.setdefault('学习率', format(learning_rate, 'f'))
        model_dict.setdefault('Hidden_size', Config["hidden_size"])
        model_dict.setdefault('Batch_size', Config["batch_size"])
        model_dict.setdefault('正确样本数', evaluator.stats_dict['correct'])
        model_dict.setdefault('错误样本数', evaluator.stats_dict['wrong'])
        model_dict.setdefault('总耗时(秒)', str(time) + '秒')
        model_dict.setdefault('文本最大长度', Config["max_length"])
        model_dict.setdefault('最后一轮准确率', acc)
        # 输出csv文件
        output_result_list.append(model_dict)
        df = pd.DataFrame(output_result_list)
        df.to_csv(Config['output_result_path'], index=False)
