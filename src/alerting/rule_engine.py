"""
规则引擎 - 基于阈值的异常告警规则
支持尖峰检测、趋势检测、持续异常检测等规则
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    """告警级别"""
    NORMAL = "正常"
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "严重"


@dataclass
class AlertRule:
    """告警规则定义"""
    name: str
    rule_type: str       # spike / trend / sustained / threshold
    parameters: Dict
    level: AlertLevel
    description: str


@dataclass
class AlertResult:
    """告警结果"""
    rule_name: str
    rule_type: str
    level: AlertLevel
    triggered: bool
    score: float         # 0~1
    message: str
    details: Dict


class RuleEngine:
    """规则引擎"""

    def __init__(
        self,
        spike_threshold: float = 3.0,
        trend_window: int = 10,
        trend_threshold: float = 0.8,
        sustained_count: int = 5,
        level_thresholds: Optional[Dict] = None,
    ):
        self.spike_threshold = spike_threshold
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.sustained_count = sustained_count
        self.level_thresholds = level_thresholds or {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }

    def evaluate(self, data: np.ndarray, anomaly_scores: Optional[np.ndarray] = None) -> List[AlertResult]:
        """评估所有规则

        参数:
            data: (n_timesteps, n_features) 原始时序数据
            anomaly_scores: (n_timesteps,) ML异常分数（可选）
        返回:
            触发的告警列表
        """
        results = []

        # 规则1: 尖峰检测
        results.append(self._check_spike(data))

        # 规则2: 趋势检测
        results.append(self._check_trend(data))

        # 规则3: 持续异常
        if anomaly_scores is not None:
            results.append(self._check_sustained(anomaly_scores))

        return results

    def _check_spike(self, data: np.ndarray) -> AlertResult:
        """尖峰检测: 值超过N倍标准差"""
        if data.ndim > 1:
            # 多变量: 检查每个特征
            max_spike_ratio = 0
            for i in range(data.shape[1]):
                col = data[:, i]
                mean = np.mean(col)
                std = np.std(col, ddof=1) + 1e-8
                ratio = np.max(np.abs(col - mean)) / std
                max_spike_ratio = max(max_spike_ratio, ratio)
        else:
            mean = np.mean(data)
            std = np.std(data, ddof=1) + 1e-8
            max_spike_ratio = np.max(np.abs(data - mean)) / std

        triggered = max_spike_ratio > self.spike_threshold
        score = min(1.0, max_spike_ratio / (self.spike_threshold * 2))

        level = self._score_to_level(score) if triggered else AlertLevel.NORMAL

        return AlertResult(
            rule_name="尖峰检测",
            rule_type="spike",
            level=level,
            triggered=triggered,
            score=score,
            message=f"检测到尖峰异常，偏差比: {max_spike_ratio:.2f} (阈值: {self.spike_threshold})",
            details={"spike_ratio": float(max_spike_ratio), "threshold": self.spike_threshold},
        )

    def _check_trend(self, data: np.ndarray) -> AlertResult:
        """趋势检测: 线性回归斜率过大"""
        if data.ndim > 1:
            max_slope = 0
            for i in range(data.shape[1]):
                col = data[-self.trend_window:, i]
                slope = self._compute_slope(col)
                # 归一化斜率
                std = np.std(data[:, i], ddof=1) + 1e-8
                normalized_slope = abs(slope) / std
                max_slope = max(max_slope, normalized_slope)
        else:
            window = data[-self.trend_window:]
            slope = self._compute_slope(window)
            std = np.std(data, ddof=1) + 1e-8
            max_slope = abs(slope) / std

        triggered = max_slope > self.trend_threshold
        score = min(1.0, max_slope / (self.trend_threshold * 2))

        level = self._score_to_level(score) if triggered else AlertLevel.NORMAL

        return AlertResult(
            rule_name="趋势检测",
            rule_type="trend",
            level=level,
            triggered=triggered,
            score=score,
            message=f"检测到异常趋势，归一化斜率: {max_slope:.2f} (阈值: {self.trend_threshold})",
            details={"normalized_slope": float(max_slope), "threshold": self.trend_threshold},
        )

    def _check_sustained(self, anomaly_scores: np.ndarray) -> AlertResult:
        """持续异常检测: 连续多个时间点异常"""
        above_threshold = anomaly_scores > self.level_thresholds.get("low", 0.3)
        # 计算最大连续异常长度
        max_run = 0
        current_run = 0
        for v in above_threshold:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        triggered = max_run >= self.sustained_count
        score = min(1.0, max_run / (self.sustained_count * 2))

        level = self._score_to_level(score) if triggered else AlertLevel.NORMAL

        return AlertResult(
            rule_name="持续异常",
            rule_type="sustained",
            level=level,
            triggered=triggered,
            score=score,
            message=f"检测到持续异常，最长连续: {max_run} (阈值: {self.sustained_count})",
            details={"max_sustained": int(max_run), "threshold": self.sustained_count},
        )

    @staticmethod
    def _compute_slope(y: np.ndarray) -> float:
        """线性回归斜率"""
        x = np.arange(len(y), dtype=float)
        n = len(x)
        if n < 2:
            return 0.0
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x ** 2) - np.sum(x) ** 2 + 1e-8
        )
        return float(slope)

    def _score_to_level(self, score: float) -> AlertLevel:
        """分数转告警级别"""
        if score >= self.level_thresholds.get("critical", 0.9):
            return AlertLevel.CRITICAL
        elif score >= self.level_thresholds.get("high", 0.7):
            return AlertLevel.HIGH
        elif score >= self.level_thresholds.get("medium", 0.5):
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
