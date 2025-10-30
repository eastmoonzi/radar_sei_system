import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from typing import Tuple

# 我们需要一个地方来保存模型，我们假设这个路径在config.yaml中定义
# 但为了快速跑通，我们先在代码里硬编码一个默认路径
# 后面我们会从config.yaml读取
DEFAULT_MODEL_DIR = "./saved_models"

def train(features: pd.DataFrame, labels: list, model_type: str, params: dict) -> Tuple[str, dict]:
    """
    使用给定的特征和标签训练一个指定类型的分类器。
    (MVP版本：暂时忽略model_type和params，只使用默认的逻辑回归)

    Args:
        features (pd.DataFrame): 用于训练的特征 (FeatureObject)。
        labels (list): 对应的真实标签列表。
        model_type (str): 要训练的模型类型 (暂时忽略)。
        params (dict): 模型的超参数 (暂时忽略)。

    Returns:
        (str, dict): 元组，包含(保存的模型路径, 训练日志)。
    """
    print(f"--- 开始训练模型 (MVP模式：使用逻辑回归) ---")
    
    # 确保模型保存目录存在
    if not os.path.exists(DEFAULT_MODEL_DIR):
        os.makedirs(DEFAULT_MODEL_DIR)
        
    model_save_path = os.path.join(DEFAULT_MODEL_DIR, "mvp_model.pkl")

    # 1. 初始化模型
    # TODO: 将来根据model_type来选择不同的模型
    model = LogisticRegression(max_iter=1000) # 增加迭代次数以保证收敛

    # 2. 训练模型
    try:
        model.fit(features, labels)
        print("模型训练完成。")
    except ValueError as e:
        print(f"训练失败：{e}")
        print("请确保你有足够的数据（至少每个类别一个样本）。")
        return None, {"status": "failed", "error": str(e)}

    # 3. 保存模型
    joblib.dump(model, model_save_path)
    print(f"模型已保存到: {model_save_path}")

    # 4. 返回模型路径和日志
    train_log = {
        "status": "success",
        "model_path": model_save_path,
        "model_type": "LogisticRegression (MVP)",
        "training_samples": len(labels),
        "classes_found": list(model.classes_)
    }
    
    return model_save_path, train_log

def predict(features: pd.DataFrame, model_path: str) -> list:
    """
    使用已加载的模型对新的特征数据进行预测。

    Args:
        features (pd.DataFrame): 待预测的特征 (FeatureObject)。
        model_path (str): 已训练模型的路径。

    Returns:
        list[PredictionObject]: 预测结果对象列表。
    """
    # 1. 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件未找到 -> {model_path}")
        return []

    # 2. 加载模型
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"模型加载失败：{e}")
        return []
        
    # 3. 执行预测
    try:
        predicted_labels = model.predict(features)
        predicted_probs = model.predict_proba(features)
        class_names = model.classes_
    except NotFittedError:
        print("错误：模型尚未训练。")
        return []
    except Exception as e:
        print(f"预测时发生错误：{e}")
        return []

    # 4. 封装成标准PredictionObject格式
    results = []
    for i in range(len(predicted_labels)):
        label = predicted_labels[i]
        probs = predicted_probs[i]
        
        # 将概率和类别名对应起来，存入字典
        prob_dict = {class_names[j]: probs[j] for j in range(len(class_names))}
        
        prediction_obj = {
            "predicted_label": label,
            "probabilities": prob_dict
        }
        results.append(prediction_obj)
        
    return results