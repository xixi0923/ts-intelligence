"""
小波特征提取 - DWT分解系数统计量
"""
import numpy as np
from typing import List


class WaveletFeatureExtractor:
    """小波变换特征提取器

    使用简化的Haar小波变换（纯NumPy实现，无需pywt依赖）
    """

    def __init__(self, level: int = 3):
        self.level = level

    @staticmethod
    def _haar_transform(signal: np.ndarray) -> tuple:
        """单层Haar小波变换

        返回:
            (approx, detail): 近似系数和细节系数
        """
        n = len(signal) // 2
        approx = np.zeros(n)
        detail = np.zeros(n)
        for i in range(n):
            approx[i] = (signal[2 * i] + signal[2 * i + 1]) / np.sqrt(2)
            detail[i] = (signal[2 * i] - signal[2 * i + 1]) / np.sqrt(2)
        return approx, detail

    def _multi_level_dwt(self, signal: np.ndarray) -> List[tuple]:
        """多层小波分解"""
        coefficients = []
        current = signal.copy()
        # 确保长度为偶数
        if len(current) % 2 != 0:
            current = current[:-1]

        for _ in range(self.level):
            if len(current) < 2:
                break
            approx, detail = self._haar_transform(current)
            coefficients.append((approx, detail))
            current = approx.copy()
            if len(current) % 2 != 0 and len(current) > 1:
                current = current[:-1]

        return coefficients

    def extract(self, window: np.ndarray) -> np.ndarray:
        """从滑动窗口提取小波特征

        参数:
            window: (window_size,) 单变量窗口
        返回:
            features: 小波特征向量
        """
        features = []
        coefficients = self._multi_level_dwt(window)

        for level_idx, (approx, detail) in enumerate(coefficients):
            # 近似系数统计量
            features.extend([
                np.mean(approx),
                np.std(approx),
                np.max(np.abs(approx)),
            ])
            # 细节系数统计量（包含异常信息的关键特征）
            features.extend([
                np.mean(detail),
                np.std(detail),
                np.max(np.abs(detail)),
                np.sum(detail ** 2),       # 细节能量
                np.mean(np.abs(detail)),   # 细节平均绝对值
            ])

        return np.array(features, dtype=float)

    def extract_batch(self, windows: np.ndarray) -> np.ndarray:
        """批量提取小波特征"""
        if windows.ndim == 3:
            all_features = []
            for feat_idx in range(windows.shape[2]):
                feat_windows = windows[:, :, feat_idx]
                feat_features = np.array([self.extract(w) for w in feat_windows])
                all_features.append(feat_features)
            return np.concatenate(all_features, axis=1)
        else:
            return np.array([self.extract(w) for w in windows])
