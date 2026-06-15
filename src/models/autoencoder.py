"""
LSTM自编码器 - 时序异常检测核心模型
通过重建误差检测异常：正常数据重建误差低，异常数据重建误差高
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LSTMAutoencoder(nn.Module):
    """LSTM自编码器

    编码器: LSTM → 潜在空间
    解码器: LSTM → 重建序列
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # 潜在空间映射
        self.fc_encode = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)

        # 解码器
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # 输出映射
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        _, (h_n, _) = self.encoder(x)
        # 取最后一层的隐藏状态
        latent = self.fc_encode(h_n[-1])
        return latent

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """从潜在空间解码重建序列"""
        # 将潜在向量映射回隐藏维度
        hidden_input = self.fc_decode(latent)  # (batch, hidden_dim)
        # 扩展到序列长度
        hidden_input = hidden_input.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        # 解码
        output, _ = self.decoder(hidden_input)
        # 映射回输入维度
        reconstructed = self.output_layer(output)
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        参数:
            x: (batch, seq_len, input_dim)
        返回:
            reconstructed: (batch, seq_len, input_dim) 重建序列
            latent: (batch, latent_dim) 潜在表示
        """
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed, latent

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """计算逐点重建误差（MSE）

        返回:
            error: (batch, seq_len, input_dim)
        """
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = (x - reconstructed) ** 2
        return error

    def get_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """计算异常分数（每个样本的标量分数）

        返回:
            scores: (batch,) numpy数组
        """
        error = self.get_reconstruction_error(x)
        # 对每个样本取平均重建误差
        scores = error.mean(dim=(1, 2)).cpu().numpy()
        return scores
