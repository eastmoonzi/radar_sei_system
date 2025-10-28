import h5py
import numpy as np

# ----------------------------------------------------
# ↓↓↓ 把你的 .h5 文件路径填在这里 ↓↓↓
FILE_PATH = "C:/radar_sei_project/SS_1021_0000_0001.h5" 
# ----------------------------------------------------

# 定义我们要侦察的关键路径
PATHS_TO_EXPLORE = [
    'IntraPulse/DATA',   # 假设这是IQ数据
    'InterPulse/LABEL',  # 假设这是标签
    'TAG/SampleRate',    # 假设这是采样率
    'TAG/CenterFreq'     # 顺便看一下中心频率
]

def print_attrs(name, obj):
    """一个辅助函数，用来打印属性"""
    print(f"    -> 属性(Attrs):")
    if not obj.attrs.items():
        print("        (无)")
        return
    for key, val in obj.attrs.items():
        print(f"        - {key}: {val}")

print(f"--- 正在对文件进行精确侦察: {FILE_PATH} ---")

try:
    with h5py.File(FILE_PATH, 'r') as f:
        
        for path in PATHS_TO_EXPLORE:
            print(f"\n--- 正在分析路径: '{path}' ---")
            
            if path not in f:
                print(f"  !! 警告: 未找到该路径 !!")
                continue

            dataset = f[path]
            
            if not isinstance(dataset, h5py.Dataset):
                print(f"  (这是一个组 (Group)，不是数据集 (Dataset))")
                continue

            # 打印数据集的详细信息
            print(f"  - 形状 (Shape): {dataset.shape}")
            print(f"  - 数据类型 (dtype): {dataset.dtype}")
            
            # 打印数据集自己的属性
            print_attrs(path, dataset)

            # 尝试读取数据本身 (根据形状决定怎么读)
            try:
                if dataset.shape == (): # 这是一个标量 (单个值)
                    data_preview = dataset[()] # 用()读取
                elif dataset.size == 1: # 只有一个元素
                    data_preview = dataset[0]
                elif dataset.ndim == 1: # 一维数组
                    data_preview = dataset[:5] # 读取前5个
                elif dataset.ndim > 1: # 多维数组
                    data_preview = dataset[0, :5] # 读取第一行的前5个
                
                print(f"  - 数据预览: {data_preview}")
                
            except Exception as e:
                print(f"  - 警告: 读取数据预览失败, {e}")


except Exception as e:
    print(f"\n读取文件失败: {e}")
    print("请确保 h5py 库已安装 (pip install h5py) 并且文件路径正确。")