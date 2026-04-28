# src/utils/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, top_k_accuracy_score


class MetricsCalculator:
    """Calculate all evaluation metrics for CAAAPT"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.tactic_true = []
        self.tactic_pred = []

    def update_binary(self, y_true, y_pred, y_pred_proba=None):
        """Update binary classification metrics"""
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        if y_pred_proba is not None:
            self.y_pred_proba.extend(y_pred_proba)

    def update_tactic(self, y_true, y_pred):
        """Update tactic classification metrics"""
        self.tactic_true.extend(y_true)
        self.tactic_pred.extend(y_pred)

    def compute_binary_metrics(self):
        """Compute precision, recall, F1 for binary classification"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='binary'
        )
        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }

    def compute_tactic_metrics(self, tactic_labels, top_k=3):
        """Compute ACC, Top3ACC, TacticACC"""
        acc = accuracy_score(self.tactic_true, self.tactic_pred) * 100

        # For top-k accuracy, need probability scores
        if hasattr(self, 'tactic_pred_proba'):
            top3acc = top_k_accuracy_score(
                self.tactic_true, self.tactic_pred_proba, k=top_k
            ) * 100
        else:
            top3acc = None

        return {
            'ACC': acc,
            'Top3ACC': top3acc if top3acc else acc,
            'TacticACC': acc
        }

    def get_confusion_stats(self):
        """Get TP, FP, TN, FN counts"""
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}