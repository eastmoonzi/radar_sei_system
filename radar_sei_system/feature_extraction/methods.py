import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, entropy
from vmdpy import VMD

def calculate_power_spectrum_features(iq_data: np.ndarray, fs: float) -> dict:
    """
    计算给定IQ信号的功率谱密度(PSD)并提取特征。
    """
    try:
        freqs, psd = welch(iq_data, fs=fs, nperseg=1024, scaling='density')
    except Exception as e:
        print(f"PSD Welch 计算失败: {e}")
        return {'psd_kurtosis': 0, 'psd_centroid': 0, 'psd_bandwidth': 0, 'psd_flatness': 0}

    psd_safe = psd[psd > 0]
    if psd_safe.size == 0:
        return {'psd_kurtosis': 0, 'psd_centroid': 0, 'psd_bandwidth': 0, 'psd_flatness': 0}

    psd_norm = psd_safe / np.sum(psd_safe)
    freqs_safe = freqs[psd > 0]

    spec_kurtosis = kurtosis(psd_norm)
    spec_centroid = np.sum(freqs_safe * psd_safe) / np.sum(psd_safe)
    spec_bandwidth = np.sqrt(np.sum(((freqs_safe - spec_centroid)**2) * psd_safe) / np.sum(psd_safe))
    spec_flatness = np.exp(np.mean(np.log(psd_safe + 1e-12))) / np.mean(psd_safe)

    features = {
        'psd_kurtosis': spec_kurtosis,
        'psd_centroid': spec_centroid,
        'psd_bandwidth': spec_bandwidth,
        'psd_flatness': spec_flatness
    }
    return features

def calculate_vmd_features(iq_data: np.ndarray, fs: float) -> dict:
    """
    使用VMD分解信号，并提取每个模态的特征。
    """
    K = 5 
    alpha = 2000 
    tau = 0.
    DC = 0
    init = 1
    tol = 1e-7

    # VMD 对信号长度很敏感，信号太长(4900万点)会内存溢出
    # 我们必须截取一段信号
    MAX_VMD_POINTS = 50000 # 截取5万个点
    if iq_data.size > MAX_VMD_POINTS:
        signal_segment = iq_data[:MAX_VMD_POINTS]
    else:
        signal_segment = iq_data

    # K个占位符特征
    placeholder_features = {}
    for k in range(K):
        placeholder_features[f'vmd_energy_{k}'] = 0
        placeholder_features[f'vmd_entropy_{k}'] = 0
    
    try:
        u, u_hat, omega = VMD(signal_segment, alpha, tau, K, DC, init, tol)
    except Exception as e:
        print(f"VMD 分解失败: {e}")
        return placeholder_features # 返回0

    features = {}
    for k in range(K):
        mode = u[k, :]
        energy = np.sum(mode**2)
        
        if energy == 0:
            shannon_entropy = 0
        else:
            p = (mode**2) / energy
            shannon_entropy = entropy(p + 1e-12)
            
        features[f'vmd_energy_{k}'] = energy
        features[f'vmd_entropy_{k}'] = shannon_entropy

    return features