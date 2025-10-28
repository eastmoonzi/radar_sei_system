import numpy as np
import os
import h5py

def load_iq_data(file_path: str) -> dict:
    """
    从项目特定的 .h5 文件中加载信号数据和元数据。

    根据侦察报告，文件结构假定如下:
    - 信号数据 (实值): 'IntraPulse/DATA' (Dataset, int32)
    - 标签 (整数): 'InterPulse/LABEL' (Dataset, int32)
    - 采样率 (整数, 单位假设为MHz): 'TAG/SampleRate' (Dataset, int32)

    Args:
        file_path (str): .h5 文件的完整路径。

    Returns:
        dict: 符合DataObject规范的字典。
    """
    
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return None
    
    try:
        with h5py.File(file_path, 'r') as f:
            
            # --- 1. 检查所有必需的路径 ---
            required_paths = ['IntraPulse/DATA', 'InterPulse/LABEL', 'TAG/SampleRate']
            for path in required_paths:
                if path not in f:
                    print(f"错误：.h5 文件结构不完整，未找到路径 -> {path}")
                    return None
            
            # =========================================================
            # --- 2. 读取信号数据 (这就是出错的地方) ---
            # 确保这两行代码在这里
            iq_data_raw = f['IntraPulse/DATA'][0, :] # [0,:] 用来解开 (1, N) 的形状
            iq_data = iq_data_raw.astype(np.float64) # 转换为浮点数
            # =========================================================

            # --- 3. 读取标签 ---
            label_int = f['InterPulse/LABEL'][0, 0]
            label_str = str(label_int) 
            
            # --- 4. 读取采样率 ---
            fs_value = f['TAG/SampleRate'][0, 0]
            
            # *** 关键假设 ***
            # 假设 '500' 的单位是 MHz
            sampling_rate_hz = fs_value * 1_000_000.0 # 500 * 1e6 = 500 MHz
            # *****************

            # --- 5. 封装成标准DataObject格式返回 ---
            # 这里的 "iq_data": iq_data 现在可以正确找到了
            data_object = {
                "iq_data": iq_data,
                "sampling_rate": sampling_rate_hz,
                "label": label_str, # 使用字符串格式的标签
                "metadata": {
                    "file_path": file_path,
                    "original_label": label_int,
                    "raw_fs_value": fs_value,
                    "is_complex": 0 # 明确标记这是实值信号
                }
            }
            
            return data_object

    except Exception as e:
        print(f"加载 .h5 文件时发生严重错误: {e}")
        print("请确保文件未损坏且 h5py 库已安装。")
        return None