"""
时序预测模型 - 基于Transformer的预测器
用于预测未来时间步，预测偏差也可作为异常信号
"""
import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple


class TimeSeriesForecaster(nn.Module):
    """时序预测模型

    编码器: 将历史窗口编码
    解码器: 预测未来horizon步
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        forecast_horizon: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon

        # 输入嵌入
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 预测头
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, forecast_horizon * input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数:
            x: (batch, seq_len, input_dim) 历史窗口
        返回:
            forecast: (batch, forecast_horizon, input_dim) 预测结果
        """
        batch_size = x.size(0)

        # 投影 + 位置编码
        h = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]

        # 编码
        encoded = self.encoder(h)  # (batch, seq_len, d_model)

        # 取最后一个时间步
        last_hidden = encoded[:, -1, :]  # (batch, d_model)

        # 预测
        forecast = self.fc_out(last_hidden)  # (batch, forecast_horizon * input_dim)
        forecast = forecast.view(batch_size, self.forecast_horizon, self.input_dim)

        return forecast

    def get_forecast_error(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> np.ndarray:
        """计算预测误差

        参数:
            x: (batch, seq_len, input_dim) 输入序列
            y: (batch, horizon, input_dim) 真实未来值
        返回:
            errors: (batch,) 预测误差
        """
        self.eval()
        with torch.no_grad():
            forecast = self.forward(x)
            error = ((forecast - y) ** 2).mean(dim=(1, 2)).cpu().numpy()
        return error
