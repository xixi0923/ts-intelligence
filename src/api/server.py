"""
FastAPI服务器 - 时序智能分析平台REST API

修复记录：
- /api/generate 改为POST JSON body（与demo_server一致）
- /api/predict 增加子模型分数返回和 model_used 标志
- /health 增加 model_ready 字段
- 添加错误处理和日志
"""
import os
import time
import logging
import numpy as np
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..data.generator import SyntheticDataGenerator
from ..data.preprocessor import Preprocessor
from ..models.detector import EnsembleAnomalyDetector
from ..pipeline.predictor import InferencePipeline
from ..pipeline.trainer import TrainingPipeline
from ..pipeline.evaluator import ModelEvaluator
from ..alerting.ml_alert import MLAlertEngine
from ..alerting.rule_engine import AlertLevel
from ...config import AppConfig

logger = logging.getLogger("ts-intelligence.server")


# ===== 请求/响应模型 =====

class TrainRequest(BaseModel):
    """训练请求"""
    scenario: str = Field(default="server_monitor", description="场景类型")
    num_epochs: int = Field(default=20, ge=1, le=100)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=32, ge=8, le=128)


class PredictRequest(BaseModel):
    """预测请求"""
    data: List[List[float]] = Field(..., description="时序数据 (n_timesteps, n_features)")
    scenario: Optional[str] = Field(default=None, description="场景类型")


class GenerateRequest(BaseModel):
    """数据生成请求"""
    scenario: str = Field(default="server_monitor")
    length: int = Field(default=1000, ge=100, le=5000)
    anomaly_ratio: float = Field(default=0.1, ge=0, le=0.5)


# ===== 全局状态 =====

app_state = {
    "model": None,
    "preprocessor": None,
    "inference_pipeline": None,
    "alert_engine": None,
    "training_history": [],
    "is_trained": False,
    "current_scenario": None,
    "threshold": None,
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
            "model_ready": app_state["inference_pipeline"] is not None,
            "scenario": app_state["current_scenario"],
        }

    @app.post("/api/train")
    async def train_model(req: TrainRequest):
        """训练模型"""
        start_time = time.time()

        try:
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

            logger.info(f"开始训练: scenario={req.scenario}, epochs={req.num_epochs}")
            data = pipeline.prepare_data(req.scenario)

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
            app_state["threshold"] = pipeline.model.threshold

            elapsed = time.time() - start_time
            result["training_time"] = elapsed
            result["scenario"] = req.scenario
            result["feature_names"] = data["feature_names"]
            result["model_used"] = True

            logger.info(f"训练完成: loss={result['final_train_loss']:.6f}, threshold={result['threshold']:.4f}")
            return result

        except Exception as e:
            logger.error(f"训练失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/predict")
    async def predict(req: PredictRequest):
        """异常检测预测"""
        if not app_state["is_trained"] or app_state["inference_pipeline"] is None:
            raise HTTPException(status_code=400, detail="模型尚未训练，请先调用 /api/train")

        data = np.array(req.data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise HTTPException(status_code=400, detail=f"数据维度错误: {data.ndim}")

        try:
            logger.info(f"推理请求: data shape={data.shape}")
            # 流式推理（使用preprocessor内部参数）
            results = app_state["inference_pipeline"].predict_stream(data)

            # 补充子模型分数
            if results and app_state["model"] is not None and app_state["preprocessor"] is not None:
                import torch
                model = app_state["model"]
                preprocessor = app_state["preprocessor"]
                inference_pipeline = app_state["inference_pipeline"]

                data_normalized = preprocessor.normalize(data)
                windows, _ = preprocessor.create_windows(data_normalized)

                if len(windows) > 0:
                    windows_tensor = torch.FloatTensor(windows).to(inference_pipeline.device)
                    with torch.no_grad():
                        lstm_scores = model.lstm_ae.get_anomaly_score(windows_tensor)
                        tf_scores = model.transformer_ae.get_anomaly_score(windows_tensor)
                    lstm_norm = model._normalize_scores(lstm_scores)
                    tf_norm = model._normalize_scores(tf_scores)
                    for i, result in enumerate(results):
                        if i < len(lstm_norm):
                            result["lstm_score"] = float(lstm_norm[i])
                        if i < len(tf_norm):
                            result["transformer_score"] = float(tf_norm[i])

            # 告警评估
            if results:
                anomaly_scores = np.array([r["anomaly_score"] for r in results])
                alerts = app_state["alert_engine"].evaluate_series(data, anomaly_scores)
            else:
                alerts = []

            logger.info(f"推理完成: {len(results)} windows, {len(alerts)} alerts")
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
                "model_used": True,
            }

        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/generate")
    async def generate_data(req: GenerateRequest):
        """生成合成数据"""
        generator = SyntheticDataGenerator(seed=42)
        sample = generator.generate(req.scenario, req.length, req.anomaly_ratio)

        return {
            "scenario": req.scenario,
            "length": req.length,
            "n_features": len(sample.feature_names),
            "feature_names": sample.feature_names,
            "anomaly_ratio": req.anomaly_ratio,
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
            "model_ready": app_state["inference_pipeline"] is not None,
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