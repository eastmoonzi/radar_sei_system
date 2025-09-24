import numpy as np
from radar_sei_system.feature_extraction.main import extract_features

# 1. 创建一个模拟的DataObject，就像B角未来会提供的那样
print("--- 1. 创建模拟数据 ---")
mock_iq = np.random.randn(8192) + 1j * np.random.randn(8192) # 8192个点
# 模拟一个单音干扰，让功率谱不那么“平坦”
t = np.arange(8192) / 1e6
mock_iq += 2 * np.exp(1j * 2 * np.pi * 100e3 * t) # 在100kHz加入一个强信号

mock_data_object = {
    "iq_data": mock_iq,
    "sampling_rate": 1e6, # 1 MHz 采样率
    "label": "mock_device_A",
    "metadata": {}
}
print("模拟数据创建成功！")

# 2. 调用你的特征提取模块
print("\n--- 2. 开始提取特征 ---")
# 我们告诉模块，我们只想用'power_spectrum'这个方法
feature_list = ['power_spectrum']
feature_object = extract_features(mock_data_object, feature_list)

print("特征提取成功！")

# 3. 查看输出结果
print("\n--- 3. 查看输出的FeatureObject ---")
print("输出类型:", type(feature_object))
print("输出内容:")
print(feature_object)