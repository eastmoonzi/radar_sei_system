# feature_extraction/main.py
import pandas as pd
# from .methods import calculate_power_spectrum_features

def extract_features(data: dict, methods: list) -> pd.DataFrame:
    feature_dict = {}
    if 'power_spectrum' in methods:
        # 调用你的特征函数
        # psd_features = calculate_power_spectrum_features(...)
        # feature_dict.update(psd_features)
        # 伪代码：
        feature_dict['psd_peak_freq'] = [0.12] 
        feature_dict['psd_energy'] = [1.25]

    # ... 其他特征方法

    return pd.DataFrame(feature_dict)