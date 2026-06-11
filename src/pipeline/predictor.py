"""
推理Pipeline - 实时/批量异常检测推理
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..models.detector import EnsembleAnomalyDetector
from ..data.preprocessor import Preprocessor


class InferencePipeline:
    """推理Pipeline"""

    def __init__(
        self,
        model: EnsembleAnomalyDetector,
        preprocessor: Preprocessor,
        device: str = "auto",
    ):
        self.model = model
        self.preprocessor = preprocessor
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_single(
        self, data: np.ndarray, return_details: bool = False
    ) -> Dict:
        """单条推理

        参数:
            data: (window_size,) 或 (window_size, n_features) 窗口数据
            return_details: 是否返回详细信息
        返回:
            推理结果字典
        """
        if data.ndim == 1:
            data = data.reshape(1, -1, 1)
        elif data.ndim == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])

        tensor = torch.FloatTensor(data).to(self.device)

        with torch.no_grad():
            anomaly_scores = self.model.get_anomaly_scores(tensor)
            anomaly_flags, point_scores = self.model.get_pointwise_anomaly(tensor)

        result = {
            "anomaly_score": float(anomaly_scores[0]),
            "is_anomaly": bool(anomaly_flags[0]),
            "threshold": self.model.threshold,
        }

        if return_details:
            result["point_scores"] = point_scores[0].tolist()
            result["lstm_score"] = float(self.model.lstm_ae.get_anomaly_score(tensor)[0])
            result["transformer_score"] = float(self.model.transformer_ae.get_anomaly_score(tensor)[0])

        return result

    def predict_batch(
        self, data: np.ndarray, return_details: bool = False
    ) -> List[Dict]:
        """批量推理

        参数:
            data: (n_samples, window_size, n_features)
        返回:
            结果列表
        """
        tensor = torch.FloatTensor(data).to(self.device)

        with torch.no_grad():
            anomaly_scores = self.model.get_anomaly_scores(tensor)
            anomaly_flags, point_scores = self.model.get_pointwise_anomaly(tensor)

        results = []
        for i in range(len(anomaly_scores)):
            result = {
                "sample_idx": i,
                "anomaly_score": float(anomaly_scores[i]),
                "is_anomaly": bool(anomaly_flags[i]),
                "threshold": self.model.threshold,
            }
            if return_details:
                result["point_scores"] = point_scores[i].tolist()
            results.append(result)

        return results

    def predict_stream(
        self,
        data: np.ndarray,
        window_size: int = 64,
        stride: int = 8,
    ) -> List[Dict]:
        """流式推理（自动创建滑动窗口）

        参数:
            data: (n_timesteps, n_features) 原始时序数据
        返回:
            窗口级推理结果列表
        """
        # 标准化
        data_normalized = self.preprocessor.normalize(data)

        # 创建窗口
        windows, _ = self.preprocessor.create_windows(data_normalized)

        if len(windows) == 0:
            return []

        # 批量推理
        results = self.predict_batch(windows, return_details=True)

        # 映射回时间轴
        for i, result in enumerate(results):
            result["window_start"] = i * stride
            result["window_end"] = i * stride + window_size

        return results
