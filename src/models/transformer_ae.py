"""
Transformer自编码器 - 基于自注意力的时序异常检测
利用注意力权重辅助异常定位
"""
import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """位置编码（正弦余弦）"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """Transformer自编码器

    编码器: Transformer Encoder → 池化 → 潜在空间
    解码器: 潜在向量 → 重复 → Transformer Decoder → 输出
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim

        # 输入嵌入
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 潜在空间
        self.fc_encode = nn.Linear(d_model, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, d_model)

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 输出映射
        self.output_proj = nn.Linear(d_model, input_dim)

        # 保存注意力权重
        self._attention_weights = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        # 输入投影 + 位置编码
        x_proj = self.input_proj(x)  # (batch, seq_len, d_model)
        x_pos = self.pos_encoder(x_proj)

        # Transformer编码
        encoded = self.transformer_encoder(x_pos)  # (batch, seq_len, d_model)

        # 全局平均池化到潜在向量
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        latent = self.fc_encode(pooled)  # (batch, latent_dim)
        return latent

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """从潜在空间解码"""
        # 映射回d_model维度
        memory_input = self.fc_decode(latent)  # (batch, d_model)
        # 扩展为序列
        memory = memory_input.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, d_model)

        # 目标序列（用零初始化，自回归风格）
        tgt = torch.zeros_like(memory)

        # Transformer解码
        decoded = self.transformer_decoder(tgt, memory)  # (batch, seq_len, d_model)

        # 输出投影
        output = self.output_proj(decoded)  # (batch, seq_len, input_dim)
        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        参数:
            x: (batch, seq_len, input_dim)
        返回:
            reconstructed: (batch, seq_len, input_dim)
            latent: (batch, latent_dim)
        """
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstructed = self.decode(latent, seq_len)
        return reconstructed, latent

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """计算重建误差"""
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            error = (x - reconstructed) ** 2
        return error

    def get_anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        """计算异常分数"""
        error = self.get_reconstruction_error(x)
        scores = error.mean(dim=(1, 2)).cpu().numpy()
        return scores
