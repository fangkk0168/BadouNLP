"""
生成验证数据：随机成功100条餐饮好评数据，100条差评数据。放入 valid.csv 中
好评：{'label':1,'review':'这家餐厅的麻婆豆腐真的太好吃了，强烈推荐！'}
差评：{'label':0,'review':'这家餐厅的 回锅肉 太失望了，味道很差。'}
"""
from config import Config
import random
import pandas as pd

# 好评模板
positive_templates = [
    "这家餐厅的{dish}真的太好吃了，强烈推荐！",
    "在{restaurant}吃饭是一次非常愉快的体验，服务也很棒。",
    "{dish}的味道太棒了，每一口都是享受。",
    "这家餐厅的环境非常优雅，食物也是一流的，特别是{dish}。",
    "我被{dish}的美味深深吸引，一定会再来。",
    "这家餐厅的{dish}是我吃过的最好吃的，强烈推荐给大家。",
    "服务非常好，{dish}也超级好吃，性价比很高。",
    "环境舒适，食物美味，特别是{dish}，让人回味无穷。",
    "这家餐厅的{dish}真的太棒了，下次还会来。",
    "强烈推荐{restaurant}的{dish}，绝对值得一试。"
]

# 差评模板
negative_templates = [
    "这家餐厅的{dish}太失望了，味道很差。",
    "在{restaurant}吃饭非常不愉快，服务也很差。",
    "{dish}的味道很一般，完全不值这个价。",
    "这家餐厅的环境很嘈杂，{dish}也很难吃。",
    "我对{dish}的味道非常失望，不会再来了。",
    "这家餐厅的{dish}太差了，完全不推荐。",
    "服务很慢，{dish}也很难吃，浪费钱。",
    "环境很差，食物也很一般，特别是{dish}，让人失望。",
    "这家餐厅的{dish}太难吃了，不会再来了。",
    "不推荐{restaurant}的{dish}，太差了。"
]

# 常见菜品和餐厅名称
dishes = ["宫保鸡丁", "鱼香肉丝", "红烧肉", "酸菜鱼", "水煮牛肉", "清蒸鲈鱼", "麻婆豆腐", "回锅肉", "糖醋排骨",
          "辣子鸡"]
restaurants = ["老北京饭庄", "川味轩", "江南春", "粤菜馆", "东北菜馆", "湘菜馆", "西餐厅", "日料店", "火锅店", "烧烤店"]

# 生成好评
positive_reviews = [
    random.choice(positive_templates).format(dish=random.choice(dishes), restaurant=random.choice(restaurants))
    for _ in range(100)
]

# 生成差评
negative_reviews = [
    random.choice(negative_templates).format(dish=random.choice(dishes), restaurant=random.choice(restaurants))
    for _ in range(100)
]

if __name__ == '__main__':
    # 打印部分好评和差评
    print("好评示例：")
    valid_list = []
    for review in positive_reviews[:100]:
        good_dict = {}
        good_dict.setdefault('label', 1)
        good_dict.setdefault('review', review)
        valid_list.append(good_dict)

    print("\n差评示例：")
    for review in negative_reviews[:100]:
        bad_dict = {}
        bad_dict.setdefault('label', 0)
        bad_dict.setdefault('review', review)
        valid_list.append(bad_dict)

    df = pd.DataFrame(valid_list)
    df.to_csv(Config['valid_data_path'], index=False)
