
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


class ModelEvaluator:
    def __init__(self, classes: list):
        self.classes = classes

    def confusion_matrix(self, y_true: list, y_pred: list) -> np.ndarray:
        return confusion_matrix(y_true, y_pred, labels=self.classes)

    def accuracy(self, y_true: list, y_pred: list) -> float:
        return accuracy_score(y_true, y_pred)

    def precision_recall_f1(self, y_true: list, y_pred: list) -> dict:

        report = classification_report(
            y_true,
            y_pred,
            labels=self.classes,
            output_dict=True,
            zero_division=0
        )

        results = {}

        for cls in self.classes:
            results[cls] = {
                'precision': round(report[cls]['precision'], 4),
                'recall':    round(report[cls]['recall'], 4),
                'f1':        round(report[cls]['f1-score'], 4),
                'support':   int(report[cls]['support'])
            }

        return results

    def macro_avg(self, report_dict: dict) -> dict:
        return {
            'precision': round(report_dict['macro avg']['precision'], 4),
            'recall':    round(report_dict['macro avg']['recall'], 4),
            'f1':        round(report_dict['macro avg']['f1-score'], 4)
        }

    def weighted_avg(self, report_dict: dict) -> dict:
        return {
            'precision': round(report_dict['weighted avg']['precision'], 4),
            'recall':    round(report_dict['weighted avg']['recall'], 4),
            'f1':        round(report_dict['weighted avg']['f1-score'], 4)
        }

    def full_report(self, y_true: list, y_pred: list) -> dict:
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=self.classes,
            output_dict=True,
            zero_division=0
        )

        acc = accuracy_score(y_true, y_pred)

        per_class = {}
        for cls in self.classes:
            per_class[cls] = {
                'precision': round(report_dict[cls]['precision'], 4),
                'recall':    round(report_dict[cls]['recall'], 4),
                'f1':        round(report_dict[cls]['f1-score'], 4),
                'support':   int(report_dict[cls]['support'])
            }

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=self.classes
        )

        macro = {
            'precision': round(report_dict['macro avg']['precision'], 4),
            'recall':    round(report_dict['macro avg']['recall'], 4),
            'f1':        round(report_dict['macro avg']['f1-score'], 4)
        }

        weighted = {
            'precision': round(report_dict['weighted avg']['precision'], 4),
            'recall':    round(report_dict['weighted avg']['recall'], 4),
            'f1':        round(report_dict['weighted avg']['f1-score'], 4)
        }

        return {
            'accuracy': round(acc, 4),
            'per_class': per_class,
            'confusion_matrix': cm,
            'macro_avg': macro,
            'weighted_avg': weighted
        }

    def print_report(self, report: dict):

        print(f"\n{'='*55}")
        print(f"  EVALUATION REPORT")
        print(f"{'='*55}")
        print(f"  Overall Accuracy : {report['accuracy']*100:.2f}%")
        print(f"{'='*55}")

        print(
            f"  {'Class':<15} "
            f"{'Precision':>10} "
            f"{'Recall':>10} "
            f"{'F1':>10} "
            f"{'Support':>10}"
        )

        print(f"  {'-'*55}")

        for cls, m in report['per_class'].items():
            print(
                f"  {cls:<15} "
                f"{m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} "
                f"{m['support']:>10}"
            )

        print(f"  {'-'*55}")

        macro = report['macro_avg']
        print(
            f"  {'macro avg':<15} "
            f"{macro['precision']:>10.4f} "
            f"{macro['recall']:>10.4f} "
            f"{macro['f1']:>10.4f}"
        )

        weighted = report['weighted_avg']
        print(
            f"  {'weighted avg':<15} "
            f"{weighted['precision']:>10.4f} "
            f"{weighted['recall']:>10.4f} "
            f"{weighted['f1']:>10.4f}"
        )

        print(f"{'='*55}\n")
