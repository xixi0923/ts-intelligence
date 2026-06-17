"""
TS-Intelligence 时序智能分析平台 - Demo服务器
一键启动：python demo_server.py
功能：合成数据生成 + 模型训练 + 异常检测 + 告警 + Web仪表盘

修复记录：
- 修复模型未被调用的根本问题：训练接口改为接收JSON body、predict端点完整链路
- 离线模式不再伪装模型已训练，避免推理时跳过真实模型
- predict_stream使用preprocessor内部参数，不再外部覆盖
- 添加Pydantic请求验证、日志记录、错误处理
"""
import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ts-intelligence")

# 内部模块
from src.data.generator import SyntheticDataGenerator
from src.data.preprocessor import Preprocessor
from src.features.statistical import StatisticalFeatureExtractor
from src.features.frequency import FrequencyFeatureExtractor
from src.features.wavelet import WaveletFeatureExtractor
from src.alerting.rule_engine import RuleEngine, AlertLevel
from src.alerting.ml_alert import MLAlertEngine

# 检测PyTorch是否可用
try:
    import torch
    from src.models.detector import EnsembleAnomalyDetector
    from src.pipeline.trainer import TrainingPipeline
    from src.pipeline.predictor import InferencePipeline
    from src.pipeline.evaluator import ModelEvaluator
    TORCH_AVAILABLE = True
    logger.info("PyTorch 已加载，支持真实模型训练与推理")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装，将使用离线演示模式（模型不会被真实调用）")


# ===== Pydantic请求模型 =====

class TrainRequest(BaseModel):
    """训练请求体"""
    scenario: str = Field(default="server_monitor", description="场景类型")
    num_epochs: int = Field(default=20, ge=1, le=100, description="训练轮数")
    learning_rate: float = Field(default=1e-3, gt=0, le=1, description="学习率")
    batch_size: int = Field(default=32, ge=8, le=128, description="批大小")


class PredictRequest(BaseModel):
    """预测请求体"""
    data: List[List[float]] = Field(..., description="时序数据，二维数组 (n_timesteps, n_features)")
    scenario: Optional[str] = Field(default=None, description="场景类型（可选，用于规则引擎）")


class GenerateRequest(BaseModel):
    """数据生成请求体"""
    scenario: str = Field(default="server_monitor", description="场景类型")
    length: int = Field(default=1000, ge=100, le=5000, description="数据长度")
    anomaly_ratio: float = Field(default=0.1, ge=0, le=0.5, description="异常比例")


# ===== 全局状态 =====
app_state = {
    "model": None,                    # EnsembleAnomalyDetector 实例
    "preprocessor": None,             # Preprocessor 实例（含fit后的mean_/std_）
    "inference_pipeline": None,       # InferencePipeline 实例
    "alert_engine": None,             # MLAlertEngine 实例
    "training_history": [],
    "is_trained": False,              # 只有真实训练成功才设为True
    "current_scenario": None,
    "feature_names": [],
    "threshold": None,
    "offline_mode": False,            # 是否处于离线演示模式
}


# ===== FastAPI应用 =====
app = FastAPI(title="TS-Intelligence 时序智能分析平台", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 - 监控仪表盘"""
    html_path = os.path.join(os.path.dirname(__file__), "src", "api", "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>TS-Intelligence</h1><p>前端页面未找到</p>"


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "torch_available": TORCH_AVAILABLE,
        "is_trained": app_state["is_trained"],
        "offline_mode": app_state["offline_mode"],
        "scenario": app_state["current_scenario"],
        "model_ready": app_state["inference_pipeline"] is not None,
    }


@app.post("/api/train")
async def train_model(request: TrainRequest):
    """训练模型 - 接收JSON body而非Query参数"""
    scenario = request.scenario
    num_epochs = request.num_epochs
    learning_rate = request.learning_rate
    batch_size = request.batch_size

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch不可用，返回离线模拟训练结果")
        app_state["offline_mode"] = True
        return _offline_train_result(scenario)

    start_time = time.time()

    try:
        # 构建训练配置
        train_config = {
            "window_size": 64,
            "stride": 8,
            "normalize_method": "zscore",
            "lstm_hidden_dim": 64,
            "lstm_num_layers": 2,
            "lstm_latent_dim": 16,
            "transformer_d_model": 64,
            "transformer_nhead": 4,
            "transformer_num_layers": 2,
            "transformer_latent_dim": 16,
            "lstm_ae_weight": 0.6,
            "transformer_ae_weight": 0.4,
        }

        pipeline = TrainingPipeline(train_config)

        # 准备数据
        logger.info(f"开始准备训练数据: 场景={scenario}")
        data = pipeline.prepare_data(scenario)
        logger.info(f"训练数据: {data['train']['data'].shape}, 特征数: {len(data['feature_names'])}")

        # 训练模型
        logger.info(f"开始训练: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
        result = pipeline.train(
            train_data=data["train"]["data"],
            val_data=data["val"]["data"],
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=5,
        )

        # 更新全局状态 - 训练成功后完整设置
        app_state["model"] = pipeline.model
        app_state["preprocessor"] = pipeline.preprocessor
        app_state["inference_pipeline"] = InferencePipeline(
            pipeline.model, pipeline.preprocessor
        )
        app_state["alert_engine"] = MLAlertEngine()
        app_state["training_history"] = pipeline.history
        app_state["is_trained"] = True          # 真实训练成功才标记
        app_state["offline_mode"] = False       # 清除离线标记
        app_state["current_scenario"] = scenario
        app_state["feature_names"] = data["feature_names"]
        app_state["threshold"] = pipeline.model.threshold

        elapsed = time.time() - start_time
        result["training_time"] = elapsed
        result["scenario"] = scenario
        result["feature_names"] = data["feature_names"]
        result["history"] = pipeline.history

        logger.info(
            f"训练完成: 场景={scenario}, 损失={result['final_train_loss']:.6f}, "
            f"阈值={result['threshold']:.4f}, 耗时={elapsed:.1f}s"
        )
        return result

    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        # 训练失败时NOT设置is_trained=True，返回错误信息而非模拟数据
        app_state["is_trained"] = False
        app_state["offline_mode"] = True
        return {
            "error": str(e),
            "is_trained": False,
            "offline_mode": True,
            "scenario": scenario,
            # 同时返回离线模拟结果供前端展示
            **_offline_train_result(scenario),
        }


@app.post("/api/predict")
async def predict(request: PredictRequest):
    """异常检测预测 - 接收Pydantic验证的JSON body"""
    if not app_state["is_trained"] and not app_state["offline_mode"]:
        raise HTTPException(status_code=400, detail="模型尚未训练，请先调用 /api/train")

    # 解析输入数据
    data = np.array(request.data, dtype=float)
    logger.info(f"收到预测请求: data shape={data.shape}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim != 2:
        raise HTTPException(status_code=400, detail=f"数据维度错误: {data.ndim}, 期望1或2维")

    # ===== 核心修复：确保模型推理链路完整 =====
    # 判断是否可以走真实模型推理
    can_use_model = (
        TORCH_AVAILABLE
        and app_state["inference_pipeline"] is not None
        and app_state["model"] is not None
        and app_state["preprocessor"] is not None
    )

    if can_use_model:
        try:
            logger.info("调用真实模型进行推理...")
            inference_pipeline = app_state["inference_pipeline"]

            # 【关键修复】predict_stream 不再外部覆盖 window_size/stride，
            # 使用 preprocessor 内部的配置（与训练时一致）
            results = inference_pipeline.predict_stream(data)

            # 【关键修复】提取模型推理的LSTM和Transformer子分数
            # predict_batch 已在 predict_stream 内部调用，results中应包含子分数
            # 但 predict_batch 的 return_details=True 只返回 point_scores
            # 需要补充 lstm_score 和 transformer_score
            if results:
                # 用模型分别获取子分数
                scores = np.array([r["anomaly_score"] for r in results])
                logger.info(f"推理完成: {len(results)} 个窗口, 最大分数={scores.max():.4f}")

                # 补充子模型分数
                preprocessor = app_state["preprocessor"]
                data_normalized = preprocessor.normalize(data)
                windows, _ = preprocessor.create_windows(data_normalized)

                if len(windows) > 0:
                    model = app_state["model"]
                    windows_tensor = torch.FloatTensor(windows).to(inference_pipeline.device)

                    with torch.no_grad():
                        lstm_scores = model.lstm_ae.get_anomaly_score(windows_tensor)
                        tf_scores = model.transformer_ae.get_anomaly_score(windows_tensor)

                    # 归一化子分数到[0,1]
                    lstm_norm = model._normalize_scores(lstm_scores)
                    tf_norm = model._normalize_scores(tf_scores)

                    for i, result in enumerate(results):
                        if i < len(lstm_norm):
                            result["lstm_score"] = float(lstm_norm[i])
                        if i < len(tf_norm):
                            result["transformer_score"] = float(tf_norm[i])

                # 告警评估
                alerts = app_state["alert_engine"].evaluate_series(data, scores)
                logger.info(f"告警评估完成: {len(alerts)} 条告警")

                return {
                    "n_windows": len(results),
                    "results": results,
                    "alerts": [
                        {
                            "timestamp": a.timestamp,
                            "level": a.alert_level.value,
                            "ml_score": a.ml_score,
                            "rule_score": a.rule_score,
                            "combined_score": a.combined_score,
                            "message": a.message,
                        }
                        for a in alerts
                    ],
                    "model_used": True,   # 标记真实模型被调用
                }
            else:
                logger.warning("推理返回0个窗口，数据长度可能不足")
                return {
                    "n_windows": 0,
                    "results": [],
                    "alerts": [],
                    "model_used": True,
                }

        except Exception as e:
            logger.error(f"模型推理失败: {e}", exc_info=True)
            # 真实推理失败，返回错误信息而非静默降级
            return {
                "error": str(e),
                "n_windows": 0,
                "results": [],
                "alerts": [],
                "model_used": False,
            }

    else:
        # 离线模式 - 返回模拟数据并明确标注
        logger.warning("模型不可用，返回离线模拟结果")
        offline_result = _offline_detect_result()
        offline_result["model_used"] = False
        offline_result["offline_mode"] = True
        return offline_result


@app.post("/api/generate")
async def generate_data(request: GenerateRequest):
    """生成合成数据 - 改为POST接收JSON body"""
    generator = SyntheticDataGenerator(seed=42)
    sample = generator.generate(request.scenario, request.length, request.anomaly_ratio)

    logger.info(f"数据生成: 场景={request.scenario}, 长度={request.length}, 特征={sample.data.shape[1]}")

    return {
        "scenario": request.scenario,
        "length": request.length,
        "n_features": len(sample.feature_names),
        "feature_names": sample.feature_names,
        "anomaly_ratio": request.anomaly_ratio,
        "data": sample.data.tolist(),
        "labels": sample.labels.tolist(),
        "timestamps": sample.timestamps.tolist(),
        "anomaly_types": sample.anomaly_types,
    }


@app.get("/api/scenarios")
async def list_scenarios():
    """列出可用场景"""
    return {
        "scenarios": [
            {"id": k, "name": v["name"], "features": v["features"]}
            for k, v in SyntheticDataGenerator.SCENARIOS.items()
        ]
    }


@app.get("/api/model/status")
async def model_status():
    """模型状态"""
    return {
        "is_trained": app_state["is_trained"],
        "offline_mode": app_state["offline_mode"],
        "torch_available": TORCH_AVAILABLE,
        "model_ready": app_state["inference_pipeline"] is not None,
        "scenario": app_state["current_scenario"],
        "threshold": app_state["threshold"],
        "training_epochs": len(app_state.get("training_history", [])),
    }


@app.get("/api/training/history")
async def training_history():
    """训练历史"""
    return {"history": app_state.get("training_history", [])}


# ===== 离线模拟 =====

def _offline_train_result(scenario: str) -> dict:
    """离线训练模拟结果 - 明确标注为模拟，不伪装模型已训练"""
    epochs = 20
    history = []
    train_loss = 0.5
    val_loss = 0.52
    for i in range(1, epochs + 1):
        train_loss *= 0.88
        val_loss *= 0.86
        history.append({"epoch": i, "train_loss": train_loss, "val_loss": val_loss})

    # 离线模式不设置 is_trained=True（防止真实推理路径被跳过）
    # 只设置 offline_mode=True
    app_state["offline_mode"] = True
    app_state["current_scenario"] = scenario
    app_state["threshold"] = 0.05
    app_state["training_history"] = history

    return {
        "epochs_trained": epochs,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "threshold": 0.05,
        "training_time": 12.5,
        "scenario": scenario,
        "history": history,
        "feature_names": ["cpu_usage", "memory_usage", "disk_io", "network_in", "network_out"],
        "offline_mode": True,           # 明确标注
        "model_used": False,            # 明确标注模型未被真实调用
    }


def _offline_detect_result() -> dict:
    """离线检测模拟结果 - 明确标注为模拟"""
    n_windows = 100
    results = []
    for i in range(n_windows):
        is_anomaly = np.random.random() < 0.12
        lstm_score = (0.5 + np.random.random() * 0.5) if is_anomaly else np.random.random() * 0.3
        tf_score = (0.4 + np.random.random() * 0.5) if is_anomaly else np.random.random() * 0.25
        ensemble = 0.6 * lstm_score + 0.4 * tf_score
        results.append({
            "sample_idx": i,
            "window_start": i * 8,
            "window_end": i * 8 + 64,
            "anomaly_score": ensemble,
            "is_anomaly": is_anomaly,
            "threshold": 0.35,
            "lstm_score": lstm_score,
            "transformer_score": tf_score,
        })

    alerts = []
    for r in results:
        if r["is_anomaly"]:
            level = "高" if r["anomaly_score"] > 0.7 else "中" if r["anomaly_score"] > 0.5 else "低"
            alerts.append({
                "timestamp": r["window_start"],
                "level": level,
                "ml_score": r["anomaly_score"],
                "rule_score": np.random.random() * 0.3,
                "combined_score": r["anomaly_score"] * 0.7 + np.random.random() * 0.3,
                "message": f"窗口{r['sample_idx']}异常分数: {r['anomaly_score']:.3f}",
            })

    return {"n_windows": n_windows, "results": results, "alerts": alerts[:8]}


# ===== 启动 =====
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  TS-Intelligence 时序智能分析平台 v1.1.0")
    print("  时序异常检测 | LSTM-AE + Transformer-AE | 规则+ML双引擎")
    print("=" * 60)
    print(f"  PyTorch: {'可用' if TORCH_AVAILABLE else '不可用（离线模式）'}")
    print(f"  访问地址: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)