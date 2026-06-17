"""
推理Pipeline - 实时/批量异常检测推理

修复记录：
- predict_stream 移除 window_size/stride 外部参数，强制使用 preprocessor 内部配置
  （与训练时一致，避免推理时参数不一致导致模型性能下降）
- predict_batch 增加 lstm_score 和 transformer_score 子分数返回
- 添加输入数据校验和日志记录
"""
import torch
import numpy as np
import logging
from typing import Dict, List, Optional
from ..models.detector import EnsembleAnomalyDetector
from ..data.preprocessor import Preprocessor

logger = logging.getLogger("ts-intelligence.predictor")


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
        logger.info(
            f"推理Pipeline初始化: device={self.device}, "
            f"window_size={preprocessor.window_size}, stride={preprocessor.stride}"
        )

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
            # 【修复】获取子模型分数并归一化
            lstm_raw = self.model.lstm_ae.get_anomaly_score(tensor)
            tf_raw = self.model.transformer_ae.get_anomaly_score(tensor)
            result["lstm_score"] = float(self.model._normalize_scores(lstm_raw)[0])
            result["transformer_score"] = float(self.model._normalize_scores(tf_raw)[0])

        return result

    def predict_batch(
        self, data: np.ndarray, return_details: bool = False
    ) -> List[Dict]:
        """批量推理

        参数:
            data: (n_samples, window_size, n_features)
            return_details: 是否返回详细信息（包含子模型分数）
        返回:
            结果列表
        """
        # 【新增】数据校验
        if data.ndim != 3:
            logger.error(f"批量推理输入维度错误: {data.ndim}, 期望3维 (n_samples, window_size, n_features)")
            return []

        tensor = torch.FloatTensor(data).to(self.device)

        with torch.no_grad():
            anomaly_scores = self.model.get_anomaly_scores(tensor)
            anomaly_flags, point_scores = self.model.get_pointwise_anomaly(tensor)

            # 【修复】获取并归一化子模型分数
            lstm_raw = self.model.lstm_ae.get_anomaly_score(tensor)
            tf_raw = self.model.transformer_ae.get_anomaly_score(tensor)
            lstm_norm = self.model._normalize_scores(lstm_raw)
            tf_norm = self.model._normalize_scores(tf_raw)

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
                # 【关键修复】返回归一化后的子模型分数
                if i < len(lstm_norm):
                    result["lstm_score"] = float(lstm_norm[i])
                if i < len(tf_norm):
                    result["transformer_score"] = float(tf_norm[i])
            results.append(result)

        logger.info(f"批量推理完成: {len(results)} 个样本, 异常数={sum(1 for r in results if r['is_anomaly'])}")
        return results

    def predict_stream(self, data: np.ndarray) -> List[Dict]:
        """流式推理（自动创建滑动窗口）

        【修复】移除 window_size 和 stride 外部参数，
        强制使用 preprocessor 内部配置，确保与训练时一致。

        参数:
            data: (n_timesteps, n_features) 原始时序数据
        返回:
            窗口级推理结果列表
        """
        # 数据校验
        if data.ndim != 2:
            logger.error(f"流式推理输入维度错误: {data.ndim}, 期望2维 (n_timesteps, n_features)")
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            else:
                return []

        # 标准化（使用训练时拟合的参数）
        data_normalized = self.preprocessor.normalize(data)

        # 创建窗口（使用 preprocessor 内部配置，与训练时一致）
        window_size = self.preprocessor.window_size
        stride = self.preprocessor.stride
        windows, _ = self.preprocessor.create_windows(data_normalized)

        if len(windows) == 0:
            logger.warning(f"数据长度{len(data)}不足创建窗口(window_size={window_size})")
            return []

        # 批量推理（return_details=True 获取子模型分数）
        results = self.predict_batch(windows, return_details=True)

        # 映射回时间轴（使用 preprocessor 的 stride）
        for i, result in enumerate(results):
            result["window_start"] = i * stride
            result["window_end"] = i * stride + window_size

        logger.info(
            f"流式推理完成: {len(results)} 个窗口, "
            f"window_size={window_size}, stride={stride}"
        )
        return results