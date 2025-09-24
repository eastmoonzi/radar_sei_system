# 示例：创建模拟DataObject
import numpy as np
mock_iq = np.random.randn(1024) + 1j * np.random.randn(1024)
mock_data_object = {
    "iq_data": mock_iq,
    "sampling_rate": 1e6,
    "label": "mock_device",
    "metadata": {}
}

