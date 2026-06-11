"""
时序数据预处理 - 标准化、滑动窗口、数据集划分
"""
import numpy as np
from typing import Optional, Tuple


class Preprocessor:
    """时序数据预处理器"""

    def __init__(
        self,
        window_size: int = 64,
        stride: int = 8,
        normalize_method: str = "zscore",
    ):
        self.window_size = window_size
        self.stride = stride
        self.normalize_method = normalize_method
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit_normalize(self, data: np.ndarray) -> np.ndarray:
        """拟合标准化参数并转换

        参数:
            data: (n_samples, n_features)
        返回:
            normalized: (n_samples, n_features)
        """
        if self.normalize_method == "zscore":
            self.mean_ = np.mean(data, axis=0)
            self.std_ = np.std(data, axis=0) + 1e-8
            return (data - self.mean_) / self.std_
        elif self.normalize_method == "minmax":
            self.min_ = np.min(data, axis=0)
            self.max_ = np.max(data, axis=0)
            range_ = self.max_ - self.min_ + 1e-8
            return (data - self.min_) / range_
        else:
            return data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """使用已拟合的参数进行标准化"""
        if self.normalize_method == "zscore" and self.mean_ is not None:
            return (data - self.mean_) / self.std_
        elif self.normalize_method == "minmax" and self.min_ is not None:
            return (data - self.min_) / (self.max_ - self.min_ + 1e-8)
        return data

    def create_windows(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """创建滑动窗口

        参数:
            data: (n_samples, n_features)
            labels: (n_samples,) 可选标签
        返回:
            windows: (n_windows, window_size, n_features)
            window_labels: (n_windows,) 窗口标签（窗口内任一为异常则标记为异常）
        """
        n_samples, n_features = data.shape
        windows = []
        window_labels = []

        for start in range(0, n_samples - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(data[start:end])
            if labels is not None:
                # 窗口内任一为异常则标记为异常
                window_labels.append(1 if np.any(labels[start:end] == 1) else 0)

        windows = np.array(windows)
        window_labels = np.array(window_labels) if window_labels else None
        return windows, window_labels

    def split_dataset(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None,
        train_ratio: float = 0.7, val_ratio: float = 0.15,
    ) -> dict:
        """划分训练/验证/测试集

        返回:
            dict: {"train": ..., "val": ..., "test": ...}
            每个值是 (data, labels) 元组
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        result = {
            "train": (data[:train_end], labels[:train_end] if labels is not None else None),
            "val": (data[train_end:val_end], labels[train_end:val_end] if labels is not None else None),
            "test": (data[val_end:], labels[val_end:] if labels is not None else None),
        }
        return result
