"""
集成异常检测器 - 融合LSTM-AE和Transformer-AE的检测能力
支持加权融合、阈值自适应、逐点异常定位
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from .autoencoder import LSTMAutoencoder
from .transformer_ae import TransformerAutoencoder


class EnsembleAnomalyDetector(nn.Module):
    """集成异常检测器

    融合策略:
    1. LSTM-AE 重建误差 (权重w1)
    2. Transformer-AE 重建误差 (权重w2)
    3. 加权融合 → 异常分数
    4. 阈值判定 → 异常/正常
    """

    def __init__(
        self,
        input_dim: int = 1,
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        lstm_latent_dim: int = 16,
        transformer_d_model: int = 64,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_latent_dim: int = 16,
        lstm_weight: float = 0.6,
        transformer_weight: float = 0.4,
    ):
        super().__init__()
        self.lstm_weight = lstm_weight
        self.transformer_weight = transformer_weight

        # 子模型
        self.lstm_ae = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            latent_dim=lstm_latent_dim,
        )
        self.transformer_ae = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            latent_dim=transformer_latent_dim,
        )

        # 阈值（训练后自动计算）
        self._threshold: Optional[float] = None
        self._score_stats: Optional[Dict] = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播

        返回:
            lstm_reconstructed: LSTM-AE重建
            transformer_reconstructed: Transformer-AE重建
            lstm_latent: LSTM潜在表示
            transformer_latent: Transformer潜在表示
        """
        lstm_recon, lstm_latent = self.lstm_ae(x)
        tf_recon, tf_latent = self.transformer_ae(x)
        return lstm_recon, tf_recon, lstm_latent, tf_latent

    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        lstm_recon, tf_recon, _, _ = self.forward(x)

        lstm_loss = nn.functional.mse_loss(lstm_recon, x)
        tf_loss = nn.functional.mse_loss(tf_recon, x)

        total_loss = (
            self.lstm_weight * lstm_loss + self.transformer_weight * tf_loss
        )

        return {
            "total_loss": total_loss,
            "lstm_loss": lstm_loss,
            "transformer_loss": tf_loss,
        }

    def get_anomaly_scores(self, x: torch.Tensor) -> np.ndarray:
        """计算集成异常分数

        返回:
            scores: (batch,) 异常分数
        """
        self.eval()
        with torch.no_grad():
            lstm_scores = self.lstm_ae.get_anomaly_score(x)
            tf_scores = self.transformer_ae.get_anomaly_score(x)

            # 归一化到[0,1]范围
            lstm_norm = self._normalize_scores(lstm_scores)
            tf_norm = self._normalize_scores(tf_scores)

            # 加权融合
            ensemble_scores = (
                self.lstm_weight * lstm_norm + self.transformer_weight * tf_norm
            )
        return ensemble_scores

    def get_pointwise_anomaly(
        self, x: torch.Tensor, threshold_percentile: float = 95.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取逐点异常定位

        返回:
            anomaly_flags: (batch,) 是否异常
            point_scores: (batch, seq_len, input_dim) 逐点异常分数
        """
        self.eval()
        with torch.no_grad():
            lstm_recon, tf_recon, _, _ = self.forward(x)

            # 逐点重建误差
            lstm_error = (x - lstm_recon) ** 2
            tf_error = (x - tf_recon) ** 2

            # 加权融合逐点误差
            point_scores = (
                self.lstm_weight * lstm_error + self.transformer_weight * tf_error
            )
            point_scores_np = point_scores.cpu().numpy()

            # 样本级异常分数
            sample_scores = point_scores.mean(dim=(1, 2)).cpu().numpy()

            # 阈值判定
            if self._threshold is None:
                self._threshold = np.percentile(sample_scores, threshold_percentile)
            anomaly_flags = (sample_scores > self._threshold).astype(int)

        return anomaly_flags, point_scores_np

    def fit_threshold(
        self, x_normal: torch.Tensor, percentile: float = 95.0
    ) -> float:
        """用正常数据拟合异常阈值

        参数:
            x_normal: 仅包含正常数据的batch
            percentile: 阈值百分位
        返回:
            threshold: 计算得到的阈值
        """
        scores = self.get_anomaly_scores(x_normal)
        self._threshold = np.percentile(scores, percentile)
        self._score_stats = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "threshold": float(self._threshold),
        }
        return self._threshold

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-Max归一化到[0,1]"""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    @property
    def score_stats(self) -> Optional[Dict]:
        return self._score_stats
