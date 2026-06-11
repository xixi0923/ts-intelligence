"""
TS-Intelligence 时序智能分析平台 - 全局配置
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DataConfig:
    """数据配置"""
    window_size: int = 64           # 滑动窗口大小
    stride: int = 8                 # 滑动步长
    train_ratio: float = 0.7        # 训练集比例
    val_ratio: float = 0.15         # 验证集比例
    test_ratio: float = 0.15        # 测试集比例
    normalize_method: str = "zscore"  # 标准化方法: zscore / minmax
    synthetic_scenarios: List[str] = field(
        default_factory=lambda: ["server_monitor", "financial", "iot_sensor", "network_traffic"]
    )


@dataclass
class FeatureConfig:
    """特征工程配置"""
    enable_statistical: bool = True     # 启用统计特征
    enable_frequency: bool = True       # 启用频域特征
    enable_wavelet: bool = True         # 启用小波特征
    fft_n_components: int = 10         # FFT保留分量数
    wavelet_type: str = "db4"          # 小波类型
    wavelet_level: int = 3             # 小波分解层数


@dataclass
class ModelConfig:
    """模型配置"""
    # LSTM自编码器
    lstm_input_dim: int = 1            # 输入维度
    lstm_hidden_dim: int = 64          # 隐藏层维度
    lstm_num_layers: int = 2           # LSTM层数
    lstm_dropout: float = 0.2          # Dropout率
    lstm_latent_dim: int = 16          # 潜在空间维度

    # Transformer自编码器
    transformer_d_model: int = 64      # 模型维度
    transformer_nhead: int = 4         # 注意力头数
    transformer_num_layers: int = 2    # Transformer层数
    transformer_dim_feedforward: int = 128  # FFN维度
    transformer_dropout: float = 0.1   # Dropout率

    # 训练参数
    learning_rate: float = 1e-3        # 学习率
    batch_size: int = 32               # 批大小
    num_epochs: int = 30               # 训练轮数
    early_stopping_patience: int = 5   # 早停耐心值
    weight_decay: float = 1e-5         # 权重衰减


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    # 重建误差阈值（自动计算时为None）
    reconstruction_threshold: Optional[float] = None
    threshold_percentile: float = 95.0    # 自动阈值百分位
    # 集成权重
    lstm_ae_weight: float = 0.6           # LSTM-AE权重
    transformer_ae_weight: float = 0.4   # Transformer-AE权重
    # 注意力异常权重
    attention_anomaly_weight: float = 0.3  # 注意力异常分数权重


@dataclass
class AlertingConfig:
    """告警配置"""
    # 告警级别阈值
    low_threshold: float = 0.3        # 低级告警阈值
    medium_threshold: float = 0.5     # 中级告警阈值
    high_threshold: float = 0.7       # 高级告警阈值
    critical_threshold: float = 0.9  # 严重告警阈值

    # 规则引擎
    enable_rule_engine: bool = True
    spike_threshold: float = 3.0      # 尖峰检测倍数（相对标准差）
    trend_window: int = 10            # 趋势检测窗口
    trend_threshold: float = 0.8      # 趋势斜率阈值

    # 告警冷却（避免重复告警）
    cooldown_seconds: int = 300       # 冷却时间（秒）


@dataclass
class APIConfig:
    """API服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_upload_size: int = 10 * 1024 * 1024  # 10MB


@dataclass
class AppConfig:
    """全局应用配置"""
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    version: str = "1.0.0"
