"""
配置文件层
"""
Config = {
    "model_type": "fast_text",
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path": "../data/chars.txt",
    "output_result_path": r"D:\code_repository\ai-study\work\week8_work.csv",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 20,
    "epoch_data_size": 200,  # 每轮训练中采样数量
    "batch_size": 32,
    "margin": 0.1,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    # 其他模型：建议使用 1e-3
    "other_learning_rate": 1e-3,
    # 如果使用bert，这个学习率需要调节的稍微低点。因为bert是一个预训练的模型有一个预训练的权重，如果设置过大就要大幅的修改学习率就会丧失bert预训练的含义
    # bert：建议使用 1e-5
    "bert_learning_rate": 1e-5,
    "positive_sample_rate": 0.5,  # 正样本比例
}
