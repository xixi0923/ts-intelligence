"""
统计特征提取 - 均值/方差/偏度/峰度/极值等
"""
import numpy as np
from typing import Dict, List


class StatisticalFeatureExtractor:
    """统计特征提取器"""

    FEATURE_NAMES = [
        "mean", "std", "min", "max", "range",
        "skewness", "kurtosis", "median",
        "q25", "q75", "iqr", "cv",
    ]

    def extract(self, window: np.ndarray) -> np.ndarray:
        """从滑动窗口提取统计特征

        参数:
            window: (window_size,) 单变量窗口
        返回:
            features: (n_features,) 统计特征向量
        """
        features = []

        # 基本统计量
        mean = np.mean(window)
        std = np.std(window)
        min_val = np.min(window)
        max_val = np.max(window)
        range_val = max_val - min_val
        median = np.median(window)

        # 偏度 (三阶矩)
        n = len(window)
        skewness = np.sum(((window - mean) / (std + 1e-8)) ** 3) / n if std > 1e-8 else 0.0

        # 峰度 (四阶矩 - 3)
        kurtosis = np.sum(((window - mean) / (std + 1e-8)) ** 4) / n - 3 if std > 1e-8 else 0.0

        # 分位数
        q25 = np.percentile(window, 25)
        q75 = np.percentile(window, 75)
        iqr = q75 - q25

        # 变异系数
        cv = std / (abs(mean) + 1e-8)

        features = np.array([
            mean, std, min_val, max_val, range_val,
            skewness, kurtosis, median,
            q25, q75, iqr, cv,
        ])
        return features

    def extract_batch(self, windows: np.ndarray) -> np.ndarray:
        """批量提取统计特征

        参数:
            windows: (n_windows, window_size) 或 (n_windows, window_size, n_features)
        返回:
            features: (n_windows, n_features * len(FEATURE_NAMES))
        """
        if windows.ndim == 3:
            # 多变量: 对每个变量分别提取
            all_features = []
            for feat_idx in range(windows.shape[2]):
                feat_windows = windows[:, :, feat_idx]
                feat_features = np.array([self.extract(w) for w in feat_windows])
                all_features.append(feat_features)
            return np.concatenate(all_features, axis=1)
        else:
            return np.array([self.extract(w) for w in windows])
