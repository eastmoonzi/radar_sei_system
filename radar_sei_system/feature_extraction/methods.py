import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis

def calculate_power_spectrum_features(iq_data: np.ndarray, fs: float) -> dict:
    """
    计算给定IQ信号的功率谱密度(PSD)并提取特征。

    Args:
        iq_data (np.ndarray): 输入的复数IQ信号。
        fs (float): 信号的采样率 (Hz)。

    Returns:
        dict: 包含多个PSD特征的字典。
    """
    # --- 步骤1: 计算功率谱密度 (PSD) ---
    # 使用Welch方法计算PSD，这是一种改进的、更稳定的周期图法。
    # nperseg: 每个段的长度，决定了频率分辨率。
    freqs, psd = welch(iq_data, fs=fs, nperseg=1024, scaling='density')

    # --- 步骤2: 从PSD中提取特征 ---

    # 特征1: 谱峭度 (Spectral Kurtosis)
    # 峭度衡量的是分布的“尖峰”程度。一个有很多杂散谐波的频谱会更“尖”。
    # 为避免log(0)错误，我们只在psd大于0的地方计算
    psd_norm = psd[psd > 0] / np.sum(psd[psd > 0])
    spec_kurtosis = kurtosis(psd_norm)

    # 特征2: 谱质心 (Spectral Centroid)
    # 频谱的“质量中心”，反映了频谱能量的中心位置。
    spec_centroid = np.sum(freqs * psd) / np.sum(psd)
    
    # 特征3: 谱带宽 (Spectral Bandwidth)
    # 衡量频谱能量的扩展范围。
    spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid)**2) * psd) / np.sum(psd))

    # 特征4: 谱平坦度 (Spectral Flatness)
    # 衡量频谱的平坦程度。值越接近1，频谱越像白噪声；值越小，频谱越有音调特性。
    # 加上一个极小值防止除以零
    spec_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / np.mean(psd)

    # --- 步骤3: 封装并返回特征 ---
    features = {
        'psd_kurtosis': spec_kurtosis,
        'psd_centroid': spec_centroid,
        'psd_bandwidth': spec_bandwidth,
        'psd_flatness': spec_flatness
    }
    
    return features