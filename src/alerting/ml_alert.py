"""
ML告警模块 - 融合模型异常分数与规则告警
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from .rule_engine import RuleEngine, AlertLevel, AlertResult


@dataclass
class CombinedAlert:
    """融合告警结果"""
    timestamp: int
    alert_level: AlertLevel
    ml_score: float               # ML异常分数 0~1
    rule_score: float             # 规则告警分数 0~1
    combined_score: float         # 融合分数 0~1
    rule_alerts: List[AlertResult]  # 触发的规则
    message: str


class MLAlertEngine:
    """ML+规则双引擎告警系统"""

    def __init__(
        self,
        ml_weight: float = 0.7,
        rule_weight: float = 0.3,
        cooldown_seconds: int = 300,
    ):
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.cooldown_seconds = cooldown_seconds
        self.rule_engine = RuleEngine()
        self._last_alert_time: Dict[str, int] = {}

    def evaluate(
        self,
        data: np.ndarray,
        ml_scores: np.ndarray,
        timestep: int = 0,
    ) -> CombinedAlert:
        """评估并融合ML和规则告警

        参数:
            data: (window_size, n_features) 时序数据窗口
            ml_scores: (n_timesteps,) ML异常分数
            timestep: 当前时间步
        返回:
            CombinedAlert: 融合告警结果
        """
        # 1. ML分数
        ml_score = float(np.max(ml_scores)) if len(ml_scores) > 0 else 0.0

        # 2. 规则评估
        rule_results = self.rule_engine.evaluate(data, ml_scores)
        triggered_rules = [r for r in rule_results if r.triggered]
        rule_score = max((r.score for r in triggered_rules), default=0.0)

        # 3. 融合
        combined_score = (
            self.ml_weight * ml_score + self.rule_weight * rule_score
        )

        # 4. 确定告警级别
        alert_level = self._score_to_level(combined_score)

        # 5. 生成消息
        if alert_level == AlertLevel.NORMAL:
            message = "系统运行正常"
        else:
            parts = []
            if ml_score > 0.5:
                parts.append(f"ML异常分数: {ml_score:.2f}")
            for r in triggered_rules:
                parts.append(f"{r.rule_name}: {r.message}")
            message = " | ".join(parts) if parts else "检测到异常"

        return CombinedAlert(
            timestamp=timestep,
            alert_level=alert_level,
            ml_score=ml_score,
            rule_score=rule_score,
            combined_score=combined_score,
            rule_alerts=triggered_rules,
            message=message,
        )

    def evaluate_series(
        self,
        data: np.ndarray,
        ml_scores: np.ndarray,
    ) -> List[CombinedAlert]:
        """对完整时序数据逐点评估

        参数:
            data: (n_timesteps, n_features)
            ml_scores: (n_timesteps,)
        返回:
            告警列表（仅包含非常告的条目）
        """
        alerts = []
        for t in range(len(ml_scores)):
            start = max(0, t - 63)
            window = data[start:t + 1]
            if len(window) < 2:
                continue

            alert = self.evaluate(window, ml_scores[start:t + 1], timestep=t)
            if alert.alert_level != AlertLevel.NORMAL:
                alerts.append(alert)

        return alerts

    @staticmethod
    def _score_to_level(score: float) -> AlertLevel:
        """分数转告警级别"""
        if score >= 0.9:
            return AlertLevel.CRITICAL
        elif score >= 0.7:
            return AlertLevel.HIGH
        elif score >= 0.5:
            return AlertLevel.MEDIUM
        elif score >= 0.3:
            return AlertLevel.LOW
        else:
            return AlertLevel.NORMAL
