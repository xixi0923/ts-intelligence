"""
训练Pipeline - 端到端训练流程
数据准备 → 特征工程 → 模型训练 → 阈值拟合 → 评估
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

from ..models.detector import EnsembleAnomalyDetector
from ..data.generator import SyntheticDataGenerator
from ..data.preprocessor import Preprocessor


class TrainingPipeline:
    """训练Pipeline"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = Preprocessor(
            window_size=config.get("window_size", 64),
            stride=config.get("stride", 8),
            normalize_method=config.get("normalize_method", "zscore"),
        )
        self.model: Optional[EnsembleAnomalyDetector] = None
        self.history: List[Dict] = []

    def prepare_data(self, scenario: str = "server_monitor") -> Dict:
        """准备训练数据"""
        # 生成合成数据
        generator = SyntheticDataGenerator(seed=42)
        length = self.config.get("data_length", 2000)
        sample = generator.generate(scenario=scenario, length=length, anomaly_ratio=0.05)

        # 标准化
        data_normalized = self.preprocessor.fit_normalize(sample.data)

        # 创建滑动窗口
        windows, window_labels = self.preprocessor.create_windows(
            data_normalized, sample.labels
        )

        # 划分数据集
        split = self.preprocessor.split_dataset(windows, window_labels)

        # 过滤训练集中的异常数据（自编码器只应使用正常数据训练）
        train_data, train_labels = split["train"]
        if train_labels is not None:
            normal_mask = train_labels == 0
            train_data = train_data[normal_mask]
            train_labels = train_labels[normal_mask]
            split["train"] = (train_data, train_labels)

        # 转为Tensor
        result = {}
        for key in ["train", "val", "test"]:
            d, l = split[key]
            result[key] = {
                "data": torch.FloatTensor(d).to(self.device),
                "labels": torch.LongTensor(l).to(self.device) if l is not None else None,
            }

        result["feature_names"] = sample.feature_names
        result["scenario"] = scenario
        return result

    def build_model(self, input_dim: int = 1) -> EnsembleAnomalyDetector:
        """构建集成检测模型"""
        model = EnsembleAnomalyDetector(
            input_dim=input_dim,
            lstm_hidden_dim=self.config.get("lstm_hidden_dim", 64),
            lstm_num_layers=self.config.get("lstm_num_layers", 2),
            lstm_latent_dim=self.config.get("lstm_latent_dim", 16),
            transformer_d_model=self.config.get("transformer_d_model", 64),
            transformer_nhead=self.config.get("transformer_nhead", 4),
            transformer_num_layers=self.config.get("transformer_num_layers", 2),
            transformer_latent_dim=self.config.get("transformer_latent_dim", 16),
            lstm_weight=self.config.get("lstm_ae_weight", 0.6),
            transformer_weight=self.config.get("transformer_ae_weight", 0.4),
        ).to(self.device)

        self.model = model
        return model

    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        patience: int = 5,
    ) -> Dict:
        """训练模型

        参数:
            train_data: (n_samples, window_size, input_dim) 训练数据
            val_data: 验证数据
        返回:
            训练结果摘要
        """
        if self.model is None:
            input_dim = train_data.shape[2] if train_data.ndim == 3 else 1
            self.build_model(input_dim)

        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        n_samples = train_data.shape[0]
        best_val_loss = float("inf")
        patience_counter = 0
        self.history = []

        for epoch in range(num_epochs):
            # 训练
            self.model.train()
            epoch_losses = []

            # 随机打乱
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = train_data[indices[start:end]]

                optimizer.zero_grad()
                losses = self.model.compute_loss(batch)
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(losses["total_loss"].item())

            avg_train_loss = np.mean(epoch_losses)

            # 验证
            val_loss = None
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = self.model.compute_loss(val_data)
                    val_loss = val_losses["total_loss"].item()
                scheduler.step(val_loss)

                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
            })

        # 用正常训练数据拟合阈值
        self.model.eval()
        with torch.no_grad():
            self.model.fit_threshold(train_data, percentile=95.0)

        return {
            "epochs_trained": len(self.history),
            "final_train_loss": self.history[-1]["train_loss"],
            "final_val_loss": self.history[-1].get("val_loss"),
            "threshold": self.model.threshold,
            "score_stats": self.model.score_stats,
        }

    def save_model(self, path: str):
        """保存模型"""
        if self.model is not None:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "threshold": self.model.threshold,
                "score_stats": self.model._score_stats,
                "config": self.config,
            }, path)

    def load_model(self, path: str, input_dim: int = 1):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.build_model(input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # 从 checkpoint 恢复阈值到 buffer
        if checkpoint.get("threshold") is not None:
            self.model._threshold.fill_(checkpoint["threshold"])
        self.model._score_stats = checkpoint.get("score_stats")
        self.config = checkpoint.get("config", {})
