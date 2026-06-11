"""
合成时序数据生成器
支持4种场景: 服务器监控、金融交易、IoT传感器、网络流量
每种场景包含正常模式 + 多种异常模式注入
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TimeSeriesSample:
    """时序数据样本"""
    data: np.ndarray               # 形状: (seq_len, n_features)
    labels: np.ndarray             # 形状: (seq_len,), 0=正常, 1=异常
    timestamps: np.ndarray        # 时间戳
    feature_names: List[str]       # 特征名列表
    scenario: str                  # 场景名
    anomaly_types: List[str]       # 注入的异常类型


class SyntheticDataGenerator:
    """合成时序数据生成器"""

    SCENARIOS = {
        "server_monitor": {
            "name": "服务器监控",
            "features": ["cpu_usage", "memory_usage", "disk_io", "network_in", "network_out"],
            "base_ranges": [(30, 60), (40, 70), (10, 40), (20, 50), (15, 45)],
        },
        "financial": {
            "name": "金融交易",
            "features": ["transaction_amount", "frequency", "unique_accounts", "avg_amount", "std_amount"],
            "base_ranges": [(100, 5000), (50, 200), (20, 80), (500, 3000), (100, 1500)],
        },
        "iot_sensor": {
            "name": "IoT传感器",
            "features": ["temperature", "vibration", "pressure", "humidity", "current"],
            "base_ranges": [(20, 35), (0.1, 2.0), (100, 120), (40, 70), (1.0, 5.0)],
        },
        "network_traffic": {
            "name": "网络流量",
            "features": ["bytes_in", "bytes_out", "packets_in", "packets_out", "connections"],
            "base_ranges": [(1e6, 5e6), (8e5, 3e6), (1e4, 5e4), (8e3, 4e4), (100, 500)],
        },
    }

    ANOMALY_TYPES = ["spike", "drift", "noise", "dropout", "oscillation"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        scenario: str = "server_monitor",
        length: int = 1000,
        anomaly_ratio: float = 0.1,
        anomaly_types: Optional[List[str]] = None,
    ) -> TimeSeriesSample:
        """生成合成时序数据"""
        if scenario not in self.SCENARIOS:
            raise ValueError(f"未知场景: {scenario}, 可选: {list(self.SCENARIOS.keys())}")

        config = self.SCENARIOS[scenario]
        n_features = len(config["features"])
        feature_names = config["features"]

        if anomaly_types is None:
            anomaly_types = self.ANOMALY_TYPES

        # 1. 生成基线数据（正常模式 + 日周期 + 噪声）
        data = np.zeros((length, n_features))
        timestamps = np.arange(length, dtype=float)

        for i, (low, high) in enumerate(config["base_ranges"]):
            baseline = (low + high) / 2
            amplitude = (high - low) / 2
            # 日周期性
            daily_cycle = amplitude * 0.3 * np.sin(2 * np.pi * timestamps / 240)
            # 周周期性
            weekly_cycle = amplitude * 0.1 * np.sin(2 * np.pi * timestamps / 1680)
            # 随机游走
            noise = self.rng.normal(0, amplitude * 0.05, length)
            walk = np.cumsum(noise)
            walk = walk - np.linspace(0, walk[-1], length)  # 回归均值
            data[:, i] = baseline + daily_cycle + weekly_cycle + walk * 0.3

        # 2. 注入异常
        labels = np.zeros(length, dtype=int)
        n_anomaly_points = int(length * anomaly_ratio)
        anomaly_segments = self._place_anomalies(length, n_anomaly_points, anomaly_types)

        for seg_start, seg_end, anom_type in anomaly_segments:
            labels[seg_start:seg_end] = 1
            for i in range(n_features):
                low, high = config["base_ranges"][i]
                amplitude = (high - low) / 2
                data[seg_start:seg_end, i] = self._inject_anomaly(
                    data[seg_start:seg_end, i], anom_type, amplitude, self.rng
                )

        return TimeSeriesSample(
            data=data,
            labels=labels,
            timestamps=timestamps,
            feature_names=feature_names,
            scenario=scenario,
            anomaly_types=[at for _, _, at in anomaly_segments],
        )

    def _place_anomalies(
        self, length: int, n_points: int, types: List[str]
    ) -> List[Tuple[int, int, str]]:
        """在时间轴上放置异常段"""
        segments = []
        remaining = n_points
        attempts = 0
        while remaining > 0 and attempts < 100:
            seg_len = min(
                self.rng.randint(5, max(6, remaining + 1)),
                remaining
            )
            seg_start = self.rng.randint(10, max(11, length - seg_len - 10))
            seg_end = seg_start + seg_len
            # 检查不与已有段重叠
            overlap = any(s < seg_end and e > seg_start for s, e, _ in segments)
            if not overlap:
                anom_type = self.rng.choice(types)
                segments.append((seg_start, seg_end, anom_type))
                remaining -= seg_len
            attempts += 1
        return segments

    @staticmethod
    def _inject_anomaly(
        segment: np.ndarray, anomaly_type: str, amplitude: float, rng: np.random.RandomState
    ) -> np.ndarray:
        """注入特定类型的异常"""
        result = segment.copy()
        length = len(result)

        if anomaly_type == "spike":
            # 尖峰异常: 突然的大幅跳升
            spike_height = amplitude * rng.uniform(2.0, 5.0)
            center = length // 2
            width = max(1, length // 4)
            for i in range(length):
                dist = abs(i - center)
                result[i] += spike_height * np.exp(-0.5 * (dist / width) ** 2)

        elif anomaly_type == "drift":
            # 漂移异常: 逐渐偏离基线
            drift_rate = amplitude * rng.uniform(0.3, 1.0) / length
            drift = np.linspace(0, drift_rate * length, length)
            result += drift * rng.choice([-1, 1])

        elif anomaly_type == "noise":
            # 噪声异常: 异常高噪声
            noise_level = amplitude * rng.uniform(0.5, 2.0)
            result += rng.normal(0, noise_level, length)

        elif anomaly_type == "dropout":
            # 信号丢失: 值突然归零或固定
            drop_point = length // 3
            result[drop_point:] = result[drop_point] * 0.1

        elif anomaly_type == "oscillation":
            # 异常振荡: 高频周期波动
            freq = rng.uniform(0.5, 2.0)
            osc_amplitude = amplitude * rng.uniform(0.5, 1.5)
            t = np.arange(length)
            result += osc_amplitude * np.sin(2 * np.pi * freq * t)

        return result

    def generate_all_scenarios(
        self, length: int = 1000, anomaly_ratio: float = 0.1
    ) -> Dict[str, TimeSeriesSample]:
        """生成所有场景的数据"""
        return {
            name: self.generate(name, length, anomaly_ratio)
            for name in self.SCENARIOS
        }
