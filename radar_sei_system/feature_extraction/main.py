import pandas as pd
import numpy as np
# 导入同目录下的methods.py中的函数
try:
    from .methods import calculate_power_spectrum_features
except ImportError:
    # 允许脚本在某些情况下被直接运行时也能工作
    from methods import calculate_power_spectrum_features

def extract_features(data: dict, methods: list) -> pd.DataFrame:
    """
    主接口函数，根据指令调用不同的特征提取方法。

    Args:
        data (dict): 符合DataObject规范的字典。
        methods (list): 要使用的特征方法名称列表。

    Returns:
        pd.DataFrame: 符合FeatureObject规范的DataFrame。
    """
    iq_data = data.get("iq_data")
    fs = data.get("sampling_rate")
    
    if iq_data is None or fs is None:
        print("错误：DataObject中缺少iq_data或sampling_rate")
        return pd.DataFrame()

    # 最终的特征字典
    all_features = {}
    
    # 根据指令调用相应的方法
    if 'power_spectrum' in methods:
        psd_features = calculate_power_spectrum_features(iq_data, fs)
        all_features.update(psd_features)
        
    # ...将来在这里添加其他特征的if分支...
    # if 'emd' in methods:
    #     emd_features = calculate_emd_features(iq_data, fs)
    #     all_features.update(emd_features)

    # 将所有提取的特征转换成Pandas DataFrame格式
    # 注意，为了符合DataFrame的格式，字典中的每个值都需要是列表或数组
    for key in all_features:
        all_features[key] = [all_features[key]] # 将单个值变成只有一个元素的列表

    if not all_features:
        print("警告：没有选择任何有效的特征提取方法。")
        return pd.DataFrame() # 如果没有选择任何方法，返回空DataFrame

    return pd.DataFrame(all_features)