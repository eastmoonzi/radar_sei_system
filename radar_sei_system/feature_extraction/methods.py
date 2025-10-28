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
    freqs, psd = welch(iq_data, fs=fs, nperseg=1024, scaling='density')

    # --- 步骤2: 从PSD中提取特征 ---
    
    # 为避免log(0)或除以0的错误，我们只在psd大于0的地方计算
    psd_safe = psd[psd > 0]
    if psd_safe.size == 0:
        # 如果信号能量太低，返回默认值
        return {
            'psd_kurtosis': 0,
            'psd_centroid': 0,
            'psd_bandwidth': 0,
            'psd_flatness': 0
        }

    psd_norm = psd_safe / np.sum(psd_safe)
    freqs_safe = freqs[psd > 0]

    spec_kurtosis = kurtosis(psd_norm)

    spec_centroid = np.sum(freqs_safe * psd_safe) / np.sum(psd_safe)
    
    spec_bandwidth = np.sqrt(np.sum(((freqs_safe - spec_centroid)**2) * psd_safe) / np.sum(psd_safe))

    spec_flatness = np.exp(np.mean(np.log(psd_safe + 1e-12))) / np.mean(psd_safe)

    # --- 步骤3: 封装并返回特征 ---
    features = {
        'psd_kurtosis': spec_kurtosis,
        'psd_centroid': spec_centroid,
        'psd_bandwidth': spec_bandwidth,
        'psd_flatness': spec_flatness
    }
    
    return features