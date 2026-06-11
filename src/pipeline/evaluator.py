"""
模型评估 - 精确率/召回率/F1/ROC-AUC等指标
"""
import numpy as np
from typing import Dict, List, Optional


class ModelEvaluator:
    """模型评估器"""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
    ) -> Dict:
        """评估模型性能

        参数:
            y_true: (n,) 真实标签
            y_pred: (n,) 预测标签
            y_scores: (n,) 异常分数（可选，用于AUC计算）
        返回:
            指标字典
        """
        # 混淆矩阵
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        # 精确率/召回率/F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 准确率
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        result = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

        # ROC-AUC (简化计算)
        if y_scores is not None:
            auc = ModelEvaluator._compute_auc(y_true, y_scores)
            result["auc"] = float(auc)

        return result

    @staticmethod
    def _compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """简化AUC计算（梯形法）"""
        # 按分数降序排列
        sorted_indices = np.argsort(-y_scores)
        y_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_list = [0.0]
        fpr_list = [0.0]
        tp = 0
        fp = 0

        for label in y_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        # 梯形法计算面积
        auc = 0.0
        for i in range(1, len(tpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

        return auc

    @staticmethod
    def find_best_threshold(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, Dict]:
        """搜索最佳阈值

        返回:
            (best_threshold, best_metrics)
        """
        best_score = -1
        best_threshold = 0.5
        best_metrics = {}

        for percentile in range(50, 100):
            threshold = np.percentile(y_scores, percentile)
            y_pred = (y_scores > threshold).astype(int)
            metrics = ModelEvaluator.evaluate(y_true, y_pred)

            score = metrics.get(metric, 0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = metrics

        return float(best_threshold), best_metrics


# 导入缺失的类型
from typing import Tuple
