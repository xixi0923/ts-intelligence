# TS-Intelligence 时序智能分析平台

基于深度学习的时序数据异常检测与智能告警平台，融合 LSTM-AE 与 Transformer-AE 双模型集成架构，提供从数据生成、特征工程、模型训练到实时告警的全链路能力。

## 特性亮点

- **双模型集成检测**：LSTM-AE + Transformer-AE 加权融合，兼顾时序依赖与全局注意力
- **多维度特征工程**：统计特征、频域特征（FFT/谱熵）、小波特征三通道联合提取
- **ML + 规则双引擎告警**：机器学习异常分数与规则引擎联合评估，支持冷却去重
- **4 种业务场景**：服务器监控、金融交易、IoT 传感器、网络流量，开箱即用
- **端到端 Pipeline**：数据生成 → 预处理 → 训练 → 评估 → 告警 → API 服务，一键启动
- **可配置阈值自适应**：基于正常数据自动拟合异常阈值，无需手动调参

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    API 服务层 (FastAPI)                    │
│         RESTful 接口 / 数据上传 / 实时推理 / 告警查询        │
├─────────────────────────────────────────────────────────┤
│                    告警引擎层                              │
│    ML 告警引擎 ── 规则引擎(尖峰/趋势/持续) ── 冷却去重       │
├─────────────────────────────────────────────────────────┤
│                    模型推理层                              │
│    LSTM-AE ── Transformer-AE ── 集成融合 ── 逐点定位       │
├─────────────────────────────────────────────────────────┤
│                    训练 Pipeline 层                       │
│    数据准备 → 特征工程 → 模型训练 → 阈值拟合 → 评估          │
├─────────────────────────────────────────────────────────┤
│                    特征工程层                              │
│    统计特征 / FFT频域特征 / Haar小波特征                     │
├─────────────────────────────────────────────────────────┤
│                    数据层                                 │
│    合成数据生成器 / 数据预处理器 / 滑动窗口 / Z-Score标准化    │
└─────────────────────────────────────────────────────────┘
```

## 技术栈

| 类别       | 技术选型                                    |
|----------|-------------------------------------------|
| 深度学习框架 | PyTorch                                   |
| 序列模型    | LSTM Autoencoder                          |
| 注意力模型   | Transformer Autoencoder (含因果掩码)         |
| 特征工程    | NumPy (FFT / Haar小波 / 统计特征)            |
| 告警引擎    | 规则引擎 + ML 融合 (冷却去重)                  |
| API 服务   | FastAPI                                   |
| 数据处理    | NumPy / Pandas                            |
| 语言       | Python 3.8+                               |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```python
from config import AppConfig
from src.pipeline.trainer import TrainingPipeline

config = AppConfig()
pipeline = TrainingPipeline({
    "window_size": 64,
    "stride": 8,
    "data_length": 2000,
    "lstm_hidden_dim": 64,
    "transformer_latent_dim": 16,
})

# 准备数据
data = pipeline.prepare_data(scenario="server_monitor")

# 训练
result = pipeline.train(
    train_data=data["train"]["data"],
    val_data=data["val"]["data"],
    num_epochs=30,
)

# 保存模型
pipeline.save_model("model.pt")
print(f"训练完成, 阈值: {result['threshold']:.4f}")
```

### 异常检测

```python
# 加载模型
pipeline.load_model("model.pt", input_dim=1)

# 计算异常分数
scores = pipeline.model.get_anomaly_scores(test_tensor)
print(f"异常分数: {scores}")
```

### 启动 API 服务

```bash
python demo_server.py
```

## API 文档

| 端点                   | 方法   | 说明              |
|----------------------|------|-------------------|
| `/health`            | GET  | 健康检查            |
| `/train`             | POST | 触发模型训练         |
| `/detect`            | POST | 上传数据执行异常检测    |
| `/alert/config`      | GET  | 获取告警配置         |
| `/alert/config`      | POST | 更新告警配置         |
| `/model/info`        | GET  | 获取模型信息与阈值     |

### 检测请求示例

```json
POST /detect
{
    "data": [[1.2, 3.4, 5.6], [2.1, 4.3, 6.5]],
    "scenario": "server_monitor"
}
```

### 检测响应示例

```json
{
    "anomaly_scores": [0.12, 0.87],
    "threshold": 0.65,
    "predictions": [0, 1],
    "alerts": [
        {
            "timestamp": 1,
            "alert_level": "高",
            "combined_score": 0.87,
            "message": "ML异常分数: 0.87 | 尖峰检测: 检测到尖峰异常"
        }
    ]
}
```

## 项目结构

```
ts-intelligence/
├── config.py                    # 全局配置（数据/特征/模型/告警/API）
├── demo_server.py               # API 服务启动入口
├── requirements.txt             # 依赖清单
├── src/
│   ├── data/
│   │   ├── generator.py         # 合成时序数据生成器（4种场景）
│   │   ├── preprocessor.py      # 数据预处理（标准化/滑动窗口/划分）
│   │   └── loader.py            # 数据加载器
│   ├── features/
│   │   ├── statistical.py       # 统计特征提取
│   │   ├── frequency.py         # 频域特征提取（FFT/谱熵/主频）
│   │   └── wavelet.py           # 小波特征提取（Haar）
│   ├── models/
│   │   ├── autoencoder.py       # LSTM 自编码器
│   │   ├── transformer_ae.py    # Transformer 自编码器（含因果掩码）
│   │   ├── detector.py          # 集成异常检测器
│   │   └── forecaster.py        # 时序预测模型
│   ├── pipeline/
│   │   ├── trainer.py           # 训练 Pipeline
│   │   ├── evaluator.py         # 模型评估（精确率/召回率/F1/AUC）
│   │   └── predictor.py         # 推理 Pipeline
│   ├── alerting/
│   │   ├── rule_engine.py       # 规则引擎（尖峰/趋势/持续异常）
│   │   └── ml_alert.py          # ML+规则融合告警引擎（含冷却）
│   └── api/
│       └── server.py            # FastAPI 服务
└── templates/                   # 前端模板
```

## 业务场景

### 1. 服务器监控
检测 CPU/内存/磁盘IO/网络等指标的异常尖峰、持续高负载、资源泄漏等模式，适用于运维监控告警。

### 2. 金融交易
识别交易金额/频率/账户数的异常波动，检测潜在欺诈交易、洗钱模式、异常交易簇。

### 3. IoT 传感器
监控温度/振动/压力/湿度/电流等传感器数据，发现设备故障前兆、信号丢失、异常振荡。

### 4. 网络流量
分析网络进出流量/包数/连接数，检测 DDoS 攻击、数据泄露、异常连接模式。

## 模型架构

### LSTM-AE + Transformer-AE 集成

```
输入序列 (batch, seq_len, input_dim)
        │
        ├─────────────────────────────────┐
        │                                 │
   LSTM 编码器                      Transformer 编码器
   (多层LSTM → 池化)              (位置编码 → 多头自注意力)
        │                                 │
   潜在空间 (16d)                   潜在空间 (16d)
        │                                 │
   LSTM 解码器                    Transformer 解码器
   (重复 → LSTM → 映射)           (因果掩码 → 交叉注意力 → 映射)
        │                                 │
   重建序列                          重建序列
        │                                 │
   重建误差 MSE                      重建误差 MSE
        │                                 │
        └────── 加权融合 (0.6 + 0.4) ──────┘
                        │
                   异常分数 [0, 1]
                        │
                   阈值判定 → 异常/正常
```

- **LSTM-AE**（权重 0.6）：擅长捕捉局部时序依赖，对渐变型异常敏感
- **Transformer-AE**（权重 0.4）：通过自注意力捕捉全局关联，对突变型异常敏感
- 归一化使用训练时统计量（buffer），保证跨批次一致性

## 配置指南

核心配置通过 `config.py` 中的 dataclass 管理：

```python
from config import AppConfig

config = AppConfig()

# 数据配置
config.data.window_size = 64       # 滑动窗口大小
config.data.stride = 8             # 滑动步长
config.data.normalize_method = "zscore"  # 标准化方法

# 模型配置
config.model.lstm_latent_dim = 16  # LSTM 潜在空间维度
config.model.transformer_latent_dim = 16  # Transformer 潜在空间维度
config.model.learning_rate = 1e-3  # 学习率

# 告警配置
config.alerting.cooldown_seconds = 300  # 告警冷却时间（秒）
config.alerting.spike_threshold = 3.0   # 尖峰检测阈值（标准差倍数）
```

## 优化变更日志

### P0 关键修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `src/pipeline/evaluator.py` | `Tuple` 导入位于文件末尾 | 移至顶部 `from typing import Dict, List, Optional, Tuple` |
| 2 | `src/models/transformer_ae.py` | 解码器缺少因果掩码，存在信息泄漏 | 使用 `generate_square_subsequent_mask` 生成因果掩码并传入解码器 |
| 3 | `src/alerting/ml_alert.py` | `cooldown_seconds` 和 `_last_alert_time` 未使用 | 按规则类型实现冷却逻辑，冷却期内同类型告警被抑制 |
| 4 | `src/models/detector.py` | 归一化使用批次级统计量，跨批次不可比 | 将 `_threshold`/`_score_min`/`_score_max` 注册为 buffer，优先使用训练时统计量 |
| 5 | `src/features/frequency.py` | `top_freqs` 计算后未添加到特征列表 | `features.extend(top_freqs)` 将主频频率加入特征向量 |

### P1 重要修复

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 6 | `config.py` | `ModelConfig` 缺少 `transformer_latent_dim`；`wavelet_type` 默认值与实现不一致 | 添加 `transformer_latent_dim: int = 16`；默认值改为 `"haar"` |
| 7 | `src/models/autoencoder.py` | `get_reconstruction_error` 调用 `self.eval()` 有副作用 | 移除 `self.eval()`，仅使用 `torch.no_grad()` |
| 8 | `src/models/transformer_ae.py` | `_attention_weights` 声明后未使用 | 移除死代码 |
| 9 | `src/pipeline/trainer.py` | 数据长度 `length=2000` 硬编码 | 从 `config.get("data_length", 2000)` 读取 |
| 10 | `src/data/generator.py` | dropout 异常 `*= 0.1` 未模拟真实信号丢失 | 改为 `result[drop_point:] = 0` |
| 11 | `src/pipeline/trainer.py` | 训练集包含 5% 异常数据，自编码器应只用正常数据 | 在 `prepare_data` 中过滤标签为 1 的训练窗口 |
| 12 | `src/alerting/rule_engine.py` | `np.std` 默认 `ddof=0`（总体标准差） | 改为 `ddof=1`（样本标准差），更适合短窗口 |

## License

MIT License
