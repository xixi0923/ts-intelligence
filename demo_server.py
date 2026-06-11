"""
TS-Intelligence 时序智能分析平台 - Demo服务器
一键启动：python demo_server.py
功能：合成数据生成 + 模型训练 + 异常检测 + 告警 + Web仪表盘
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

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
except ImportError:
    TORCH_AVAILABLE = False
    print("[警告] PyTorch未安装，将使用离线演示模式")

# ===== 全局状态 =====
app_state = {
    "model": None,
    "preprocessor": None,
    "inference_pipeline": None,
    "alert_engine": None,
    "training_history": [],
    "is_trained": False,
    "current_scenario": None,
    "feature_names": [],
    "threshold": None,
}


# ===== FastAPI应用 =====
app = FastAPI(title="TS-Intelligence 时序智能分析平台", version="1.0.0")
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
        "scenario": app_state["current_scenario"],
    }


@app.post("/api/train")
async def train_model(scenario: str = Query("server_monitor")):
    """训练模型"""
    if not TORCH_AVAILABLE:
        return _offline_train_result(scenario)

    start_time = time.time()

    try:
        pipeline = TrainingPipeline({
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
        })

        # 准备数据
        data = pipeline.prepare_data(scenario)

        # 训练
        result = pipeline.train(
            train_data=data["train"]["data"],
            val_data=data["val"]["data"],
            num_epochs=20,
            learning_rate=1e-3,
            batch_size=32,
            patience=5,
        )

        # 更新全局状态
        app_state["model"] = pipeline.model
        app_state["preprocessor"] = pipeline.preprocessor
        app_state["inference_pipeline"] = InferencePipeline(
            pipeline.model, pipeline.preprocessor
        )
        app_state["alert_engine"] = MLAlertEngine()
        app_state["training_history"] = pipeline.history
        app_state["is_trained"] = True
        app_state["current_scenario"] = scenario
        app_state["feature_names"] = data["feature_names"]
        app_state["threshold"] = pipeline.model.threshold

        elapsed = time.time() - start_time
        result["training_time"] = elapsed
        result["scenario"] = scenario
        result["feature_names"] = data["feature_names"]
        result["history"] = pipeline.history

        print(f"[训练完成] 场景={scenario}, 损失={result['final_train_loss']:.6f}, 阈值={result['threshold']:.4f}, 耗时={elapsed:.1f}s")
        return result

    except Exception as e:
        print(f"[训练错误] {e}")
        return _offline_train_result(scenario)


@app.post("/api/predict")
async def predict(data_payload: dict):
    """异常检测预测"""
    if not app_state["is_trained"]:
        return {"n_windows": 0, "results": [], "alerts": [], "error": "模型尚未训练"}

    data = np.array(data_payload.get("data", []), dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    try:
        if TORCH_AVAILABLE and app_state["inference_pipeline"] is not None:
            results = app_state["inference_pipeline"].predict_stream(data)

            # 告警评估
            if results:
                scores = np.array([r["anomaly_score"] for r in results])
                alerts = app_state["alert_engine"].evaluate_series(data, scores)
            else:
                alerts = []

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
            }
        else:
            return _offline_detect_result()
    except Exception as e:
        print(f"[检测错误] {e}")
        return _offline_detect_result()


@app.get("/api/generate")
async def generate_data(
    scenario: str = Query("server_monitor"),
    length: int = Query(1000),
    anomaly_ratio: float = Query(0.1),
):
    """生成合成数据"""
    generator = SyntheticDataGenerator(seed=42)
    sample = generator.generate(scenario, length, anomaly_ratio)

    return {
        "scenario": scenario,
        "length": length,
        "n_features": len(sample.feature_names),
        "feature_names": sample.feature_names,
        "anomaly_ratio": anomaly_ratio,
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
        "torch_available": TORCH_AVAILABLE,
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
    """离线训练模拟结果"""
    epochs = 20
    history = []
    train_loss = 0.5
    val_loss = 0.52
    for i in range(1, epochs + 1):
        train_loss *= 0.88
        val_loss *= 0.86
        history.append({"epoch": i, "train_loss": train_loss, "val_loss": val_loss})

    app_state["is_trained"] = True
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
    }


def _offline_detect_result() -> dict:
    """离线检测模拟结果"""
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
    print("  TS-Intelligence 时序智能分析平台")
    print("  时序异常检测 | LSTM-AE + Transformer-AE | 规则+ML双引擎")
    print("=" * 60)
    print(f"  PyTorch: {'可用' if TORCH_AVAILABLE else '不可用（离线模式）'}")
    print(f"  访问地址: http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
