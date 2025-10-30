import pandas as pd
import numpy as np

# 导入同目录下的methods.py中的函数
try:
    from .methods import calculate_power_spectrum_features, calculate_vmd_features
except ImportError:
    # 允许脚本在某些情况下被直接运行时也能工作
    from methods import calculate_power_spectrum_features, calculate_vmd_features

def extract_features(data: dict, methods: list) -> pd.DataFrame:
    """
    主接口函数，根据指令调用不同的特征提取方法。
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
        if psd_features: # 确保返回的不是None
            all_features.update(psd_features)
        
    if 'vmd' in methods:
        vmd_features = calculate_vmd_features(iq_data, fs)
        if vmd_features: # 确保返回的不是None
            all_features.update(vmd_features)

    # ...可以继续添加其他方法的调用

    # 检查：如果两个都调用了，但all_features还是空的
    if not all_features:
        # 这就是你看到的日志
        print("警告：没有选择任何有效的特征提取方法。(或所有方法均提取失败)") 
        return pd.DataFrame()

    # 将所有提取的特征转换成Pandas DataFrame格式
    for key in all_features:
        all_features[key] = [all_features[key]] # 将单个值变成只有一个元素的列表

    return pd.DataFrame(all_features)