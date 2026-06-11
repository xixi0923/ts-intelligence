"""
数据加载器 - 支持CSV文件和内存数据加载
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path


class DataLoader:
    """时序数据加载器"""

    @staticmethod
    def from_csv(
        filepath: str,
        time_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        label_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """从CSV文件加载数据

        返回:
            data: (n_samples, n_features) 数组
            labels: (n_samples,) 标签数组，无标签时为None
            feature_names: 特征名列表
        """
        df = pd.read_csv(filepath)

        if feature_cols is not None:
            features = df[feature_cols].values
            names = feature_cols
        else:
            # 排除时间列和标签列
            exclude = set()
            if time_col:
                exclude.add(time_col)
            if label_col:
                exclude.add(label_col)
            feature_df = df.drop(columns=list(exclude))
            features = feature_df.values
            names = feature_df.columns.tolist()

        labels = df[label_col].values if label_col else None
        return features, labels, names

    @staticmethod
    def from_array(
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """从numpy数组加载数据"""
        n_features = data.shape[1] if data.ndim > 1 else 1
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        return data, feature_names
