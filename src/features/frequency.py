"""
频域特征提取 - FFT/谱熵/主频等
"""
import numpy as np
from typing import Optional


class FrequencyFeatureExtractor:
    """频域特征提取器"""

    def __init__(self, n_components: int = 10, sample_rate: float = 1.0):
        self.n_components = n_components
        self.sample_rate = sample_rate

    def extract(self, window: np.ndarray) -> np.ndarray:
        """从滑动窗口提取频域特征

        参数:
            window: (window_size,) 单变量窗口
        返回:
            features: 频域特征向量
        """
        features = []

        # FFT变换
        fft_vals = np.fft.rfft(window)
        fft_magnitude = np.abs(fft_vals)
        fft_phase = np.angle(fft_vals)

        # 1. 主频分量幅度（前N个）
        top_magnitudes = fft_magnitude[1:self.n_components + 1]  # 跳过直流分量
        if len(top_magnitudes) < self.n_components:
            top_magnitudes = np.pad(top_magnitudes, (0, self.n_components - len(top_magnitudes)))
        features.extend(top_magnitudes)

        # 2. 主频频率
        freqs = np.fft.rfftfreq(len(window), d=1.0 / self.sample_rate)
        top_freqs = freqs[1:self.n_components + 1]
        if len(top_freqs) < self.n_components:
            top_freqs = np.pad(top_freqs, (0, self.n_components - len(top_freqs)))

        # 3. 主频位置（最大幅度的频率）
        dominant_freq_idx = np.argmax(fft_magnitude[1:]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        features.append(dominant_freq)

        # 4. 谱熵（衡量频率分布的均匀性）
        power_spectrum = fft_magnitude[1:] ** 2
        total_power = np.sum(power_spectrum) + 1e-8
        psd_normalized = power_spectrum / total_power
        psd_normalized = psd_normalized[psd_normalized > 0]
        spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))
        features.append(spectral_entropy)

        # 5. 总功率
        total_power_val = np.sum(power_spectrum)
        features.append(total_power_val)

        # 6. 频谱重心
        if len(freqs[1:]) > 0 and total_power_val > 1e-8:
            spectral_centroid = np.sum(freqs[1:] * power_spectrum) / total_power_val
        else:
            spectral_centroid = 0.0
        features.append(spectral_centroid)

        # 7. 频谱带宽
        if total_power_val > 1e-8:
            spectral_bandwidth = np.sqrt(
                np.sum(((freqs[1:] - spectral_centroid) ** 2) * power_spectrum) / total_power_val
            )
        else:
            spectral_bandwidth = 0.0
        features.append(spectral_bandwidth)

        return np.array(features, dtype=float)

    def extract_batch(self, windows: np.ndarray) -> np.ndarray:
        """批量提取频域特征"""
        if windows.ndim == 3:
            all_features = []
            for feat_idx in range(windows.shape[2]):
                feat_windows = windows[:, :, feat_idx]
                feat_features = np.array([self.extract(w) for w in feat_windows])
                all_features.append(feat_features)
            return np.concatenate(all_features, axis=1)
        else:
            return np.array([self.extract(w) for w in windows])
