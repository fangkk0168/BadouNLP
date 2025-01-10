"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "model_type": "bert",
    "train_data_path": r"D:\code_repository\ai-study\work\train.csv",
    "valid_data_path": r"D:\code_repository\ai-study\work\valid.csv",
    "vocab_path": "./chars.txt",
    "max_length": 25,
    "hidden_size": 256,
    "class_num": 2,
    # 这里指定的是bert训练层数。如果任务比较简单这个层数可以设置低点，层数越高，消耗的资源越多。
    "num_layers": 1,
    "batch_size": 128,
    "epoch": 15,
    "pooling_style": "max",
    "optimizer": "adam",
    # 其他模型：建议使用 1e-3
    "other_learning_rate": 1e-3,
    # 如果使用bert，这个学习率需要调节的稍微低点。因为bert是一个预训练的模型有一个预训练的权重，如果设置过大就要大幅的修改学习率就会丧失bert预训练的含义
    # bert：建议使用 1e-5
    "bert_learning_rate": 1e-5,
    "pretrain_model_path": r'D:\code_repository\ai-study\bert-base-chinese',
    "seed": 987
}
