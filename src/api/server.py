"""
FastAPI服务器 - 时序智能分析平台REST API
"""
import os
import json
import time
import numpy as np
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..data.generator import SyntheticDataGenerator
from ..data.preprocessor import Preprocessor
from ..models.detector import EnsembleAnomalyDetector
from ..pipeline.predictor import InferencePipeline
from ..pipeline.trainer import TrainingPipeline
from ..pipeline.evaluator import ModelEvaluator
from ..alerting.ml_alert import MLAlertEngine
from ..alerting.rule_engine import AlertLevel
from ...config import AppConfig


# ===== 请求/响应模型 =====

class TrainRequest(BaseModel):
    """训练请求"""
    scenario: str = "server_monitor"
    num_epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 32


class PredictRequest(BaseModel):
    """预测请求"""
    data: List[List[float]]   # (n_timesteps, n_features)
    scenario: Optional[str] = None


class AnomalyResult(BaseModel):
    """异常检测结果"""
    anomaly_score: float
    is_anomaly: bool
    threshold: Optional[float]
    alert_level: str
    message: str


# ===== 全局状态 =====

app_state = {
    "model": None,
    "preprocessor": None,
    "inference_pipeline": None,
    "alert_engine": None,
    "training_history": [],
    "is_trained": False,
    "current_scenario": None,
}


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """创建FastAPI应用"""
    if config is None:
        config = AppConfig()

    app = FastAPI(
        title="TS-Intelligence 时序智能分析平台",
        description="基于深度学习的时序异常检测与告警系统",
        version=config.version,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 静态文件
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # ===== 路由 =====

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """首页 - 监控仪表盘"""
        html_path = os.path.join(static_dir, "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>TS-Intelligence 时序智能分析平台</h1><p>前端页面未找到</p>"

    @app.get("/health")
    async def health():
        """健康检查"""
        return {
            "status": "ok",
            "is_trained": app_state["is_trained"],
            "scenario": app_state["current_scenario"],
        }

    @app.post("/api/train")
    async def train_model(req: TrainRequest):
        """训练模型"""
        start_time = time.time()

        pipeline = TrainingPipeline({
            "window_size": config.data.window_size,
            "stride": config.data.stride,
            "normalize_method": config.data.normalize_method,
            "lstm_hidden_dim": config.model.lstm_hidden_dim,
            "lstm_num_layers": config.model.lstm_num_layers,
            "lstm_latent_dim": config.model.lstm_latent_dim,
            "transformer_d_model": config.model.transformer_d_model,
            "transformer_nhead": config.model.transformer_nhead,
            "transformer_num_layers": config.model.transformer_num_layers,
            "transformer_latent_dim": config.model.transformer_latent_dim,
            "lstm_ae_weight": config.anomaly.lstm_ae_weight,
            "transformer_ae_weight": config.anomaly.transformer_ae_weight,
        })

        # 准备数据
        data = pipeline.prepare_data(req.scenario)

        # 训练
        result = pipeline.train(
            train_data=data["train"]["data"],
            val_data=data["val"]["data"],
            num_epochs=req.num_epochs,
            learning_rate=req.learning_rate,
            batch_size=req.batch_size,
            patience=config.model.early_stopping_patience,
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
        app_state["current_scenario"] = req.scenario

        elapsed = time.time() - start_time
        result["training_time"] = elapsed
        result["scenario"] = req.scenario
        result["feature_names"] = data["feature_names"]

        return result

    @app.post("/api/predict")
    async def predict(req: PredictRequest):
        """异常检测预测"""
        if not app_state["is_trained"] or app_state["inference_pipeline"] is None:
            raise HTTPException(status_code=400, detail="模型尚未训练，请先调用 /api/train")

        data = np.array(req.data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # 流式推理
        results = app_state["inference_pipeline"].predict_stream(data)

        # 告警评估
        if results:
            anomaly_scores = np.array([r["anomaly_score"] for r in results])
            alerts = app_state["alert_engine"].evaluate_series(data, anomaly_scores)
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

    @app.post("/api/generate")
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
        model = app_state.get("model")
        return {
            "is_trained": app_state["is_trained"],
            "scenario": app_state["current_scenario"],
            "threshold": model.threshold if model else None,
            "score_stats": model.score_stats if model else None,
            "training_epochs": len(app_state.get("training_history", [])),
        }

    @app.get("/api/training/history")
    async def training_history():
        """训练历史"""
        return {"history": app_state.get("training_history", [])}

    return app
