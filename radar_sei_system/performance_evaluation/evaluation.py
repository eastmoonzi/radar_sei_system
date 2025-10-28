import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(predictions: list, true_labels: list) -> dict:
    """
    对一批预测结果进行全面的性能评估。
    (MVP版本：只实现准确率和混淆矩阵)

    Args:
        predictions (list[PredictionObject]): 模型输出的预测结果列表。
        true_labels (list): 对应的真实标签列表。

    Returns:
        dict: 包含所有评估指标的标准评估对象 (EvaluationObject)。
    """
    
    # 1. 从PredictionObject列表中提取预测的标签
    predicted_labels = [p["predicted_label"] for p in predictions]

    # 2. 检查输入是否有效
    if len(predicted_labels) == 0 or len(true_labels) == 0 or len(predicted_labels) != len(true_labels):
        print("评估错误：预测列表或真实标签列表为空，或两者长度不匹配。")
        return {
            "accuracy": 0.0,
            "confusion_matrix": None,
            "status": "error: input mismatch"
        }
        
    # 3. 计算核心指标
    try:
        # 计算准确率
        acc = accuracy_score(true_labels, predicted_labels)
        
        # 计算混淆矩阵
        # 获取所有可能的标签，确保混淆矩阵的维度正确
        all_labels = sorted(list(set(true_labels) | set(predicted_labels)))
        cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

        # 4. 封装成标准EvaluationObject格式
        evaluation_result = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "status": "success",
            "total_samples": len(true_labels),
            "labels_in_matrix": all_labels
            # ... 将来在这里添加其他评估指标 ...
            # "class_separation_score": 0.0,
            # "class_aggregation_score": 0.0,
        }
        
        return evaluation_result
        
    except Exception as e:
        print(f"评估时发生错误: {e}")
        return {
            "accuracy": 0.0,
            "confusion_matrix": None,
            "status": f"error: {e}"
        }