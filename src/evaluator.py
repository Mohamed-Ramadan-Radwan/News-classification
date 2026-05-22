"""
Model Evaluator - all metrics computed from scratch (no sklearn.metrics)
"""

import numpy as np
from collections import defaultdict


class ModelEvaluator:
    def __init__(self, classes: list):
        self.classes = classes

    def confusion_matrix(self, y_true: list, y_pred: list) -> np.ndarray:
        """
        Build confusion matrix from scratch.
        Rows = actual, Columns = predicted
        """
        n = len(self.classes)
        cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        matrix = np.zeros((n, n), dtype=int)

        for true, pred in zip(y_true, y_pred):
            i = cls_to_idx[true]
            j = cls_to_idx[pred]
            matrix[i][j] += 1

        return matrix

    def accuracy(self, y_true: list, y_pred: list) -> float:
        """Accuracy = correct predictions / total predictions"""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    def precision_recall_f1(self, y_true: list, y_pred: list) -> dict:
        """
        Compute per-class Precision, Recall, F1 from scratch.
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * P * R / (P + R)
        """
        results = {}
        for cls in self.classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            results[cls] = {
                'precision': round(precision, 4),
                'recall':    round(recall, 4),
                'f1':        round(f1, 4),
                'support':   sum(1 for t in y_true if t == cls)
            }
        return results

    def macro_avg(self, per_class: dict) -> dict:
        """Macro average = unweighted mean across classes"""
        keys = ['precision', 'recall', 'f1']
        return {k: round(np.mean([per_class[c][k] for c in self.classes]), 4)
                for k in keys}

    def weighted_avg(self, per_class: dict, total: int) -> dict:
        """Weighted average = weighted by support (class size)"""
        keys = ['precision', 'recall', 'f1']
        result = {}
        for k in keys:
            weighted_sum = sum(
                per_class[c][k] * per_class[c]['support']
                for c in self.classes
            )
            result[k] = round(weighted_sum / total, 4)
        return result

    def full_report(self, y_true: list, y_pred: list) -> dict:
        """Complete evaluation report"""
        acc = self.accuracy(y_true, y_pred)
        per_class = self.precision_recall_f1(y_true, y_pred)
        cm = self.confusion_matrix(y_true, y_pred)
        macro = self.macro_avg(per_class)
        weighted = self.weighted_avg(per_class, len(y_true))

        return {
            'accuracy': round(acc, 4),
            'per_class': per_class,
            'confusion_matrix': cm,
            'macro_avg': macro,
            'weighted_avg': weighted
        }

    def print_report(self, report: dict):
        """Pretty-print the evaluation report"""
        print(f"\n{'='*55}")
        print(f"  EVALUATION REPORT")
        print(f"{'='*55}")
        print(f"  Overall Accuracy : {report['accuracy']*100:.2f}%")
        print(f"{'='*55}")
        print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-'*55}")
        for cls, m in report['per_class'].items():
            print(f"  {cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
        print(f"  {'-'*55}")
        macro = report['macro_avg']
        print(f"  {'macro avg':<15} {macro['precision']:>10.4f} {macro['recall']:>10.4f} {macro['f1']:>10.4f}")
        weighted = report['weighted_avg']
        print(f"  {'weighted avg':<15} {weighted['precision']:>10.4f} {weighted['recall']:>10.4f} {weighted['f1']:>10.4f}")
        print(f"{'='*55}\n")
