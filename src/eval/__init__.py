"""Evaluation metrics and utilities for phishing detection."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve
)


class PhishingEvaluator:
    """Comprehensive evaluation for phishing detection models."""
    
    def __init__(self, k_values: List[int] = [10, 50, 100], target_precision: float = 0.9):
        """Initialize evaluator with configuration.
        
        Args:
            k_values: List of K values for precision@K calculation
            target_precision: Target precision for recall calculation
        """
        self.k_values = k_values
        self.target_precision = target_precision
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))
            
            # Precision@K metrics
            metrics.update(self._calculate_precision_at_k(y_true, y_proba))
            
            # Recall at target precision
            metrics.update(self._calculate_recall_at_precision(y_true, y_proba))
            
            # Alert volume metrics
            metrics.update(self._calculate_alert_metrics(y_true, y_proba))
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of basic metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _calculate_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate probability-based metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of probability metrics
        """
        # Ensure y_proba has correct shape
        if y_proba.ndim == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true, y_proba_positive)
        except ValueError:
            roc_auc = 0.0
        
        # Precision-Recall AUC
        try:
            aucpr = average_precision_score(y_true, y_proba_positive)
        except ValueError:
            aucpr = 0.0
        
        return {
            'roc_auc': roc_auc,
            'aucpr': aucpr
        }
    
    def _calculate_precision_at_k(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate precision@K metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of precision@K metrics
        """
        # Ensure y_proba has correct shape
        if y_proba.ndim == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(y_proba_positive)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        precision_at_k = {}
        for k in self.k_values:
            if k <= len(sorted_labels):
                top_k_labels = sorted_labels[:k]
                precision_at_k[f'precision_at_{k}'] = np.mean(top_k_labels)
            else:
                precision_at_k[f'precision_at_{k}'] = np.mean(sorted_labels)
        
        return precision_at_k
    
    def _calculate_recall_at_precision(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate recall at target precision.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of recall at precision metrics
        """
        # Ensure y_proba has correct shape
        if y_proba.ndim == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba_positive)
        
        # Find threshold that achieves target precision
        target_precision_indices = np.where(precision >= self.target_precision)[0]
        
        if len(target_precision_indices) > 0:
            # Use the highest recall that achieves target precision
            max_recall_idx = np.argmax(recall[target_precision_indices])
            recall_at_precision = recall[target_precision_indices[max_recall_idx]]
            threshold_at_precision = thresholds[target_precision_indices[max_recall_idx]]
        else:
            recall_at_precision = 0.0
            threshold_at_precision = 1.0
        
        return {
            f'recall_at_precision_{self.target_precision}': recall_at_precision,
            f'threshold_at_precision_{self.target_precision}': threshold_at_precision
        }
    
    def _calculate_alert_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate alert volume and workload metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of alert metrics
        """
        # Ensure y_proba has correct shape
        if y_proba.ndim == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        # Calculate alerts at different thresholds
        thresholds = [0.5, 0.7, 0.9]
        alert_metrics = {}
        
        for threshold in thresholds:
            alerts = (y_proba_positive >= threshold).astype(int)
            alert_rate = np.mean(alerts)
            true_alerts = np.sum((alerts == 1) & (y_true == 1))
            false_alerts = np.sum((alerts == 1) & (y_true == 0))
            
            alert_metrics.update({
                f'alert_rate_threshold_{threshold}': alert_rate,
                f'true_alerts_threshold_{threshold}': int(true_alerts),
                f'false_alerts_threshold_{threshold}': int(false_alerts),
                f'false_positive_rate_threshold_{threshold}': false_alerts / max(np.sum(y_true == 0), 1)
            })
        
        return alert_metrics
    
    def calculate_false_positive_rate_at_tpr(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                          target_tpr: float = 0.95) -> float:
        """Calculate false positive rate at target true positive rate.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            target_tpr: Target true positive rate
            
        Returns:
            False positive rate at target TPR
        """
        # Ensure y_proba has correct shape
        if y_proba.ndim == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_positive)
        
        # Find threshold that achieves target TPR
        target_tpr_indices = np.where(tpr >= target_tpr)[0]
        
        if len(target_tpr_indices) > 0:
            # Use the lowest FPR that achieves target TPR
            min_fpr_idx = np.argmin(fpr[target_tpr_indices])
            fpr_at_tpr = fpr[target_tpr_indices[min_fpr_idx]]
        else:
            fpr_at_tpr = 1.0
        
        return fpr_at_tpr
    
    def create_leaderboard(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create a leaderboard from multiple model results.
        
        Args:
            results: Dictionary mapping model names to their evaluation results
            
        Returns:
            Leaderboard with sorted results
        """
        leaderboard = {}
        
        # Primary metrics for ranking
        primary_metrics = ['aucpr', 'f1_score', 'precision_at_10', 'precision_at_50']
        
        for metric in primary_metrics:
            if metric in primary_metrics:
                # Sort models by this metric (descending)
                sorted_models = sorted(
                    results.items(),
                    key=lambda x: x[1].get(metric, 0),
                    reverse=True
                )
                
                leaderboard[metric] = [
                    {
                        'model': model_name,
                        'score': model_results.get(metric, 0),
                        'rank': i + 1
                    }
                    for i, (model_name, model_results) in enumerate(sorted_models)
                ]
        
        return leaderboard
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 model_name: str = "Model") -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        metrics = self.evaluate_model(y_true, y_pred, y_proba)
        
        report = f"\n{'='*60}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*60}\n\n"
        
        # Basic metrics
        report += "BASIC METRICS:\n"
        report += f"  Accuracy:  {metrics['accuracy']:.4f}\n"
        report += f"  Precision: {metrics['precision']:.4f}\n"
        report += f"  Recall:    {metrics['recall']:.4f}\n"
        report += f"  F1-Score:  {metrics['f1_score']:.4f}\n\n"
        
        # Probability metrics
        if y_proba is not None:
            report += "PROBABILITY METRICS:\n"
            report += f"  ROC AUC:   {metrics['roc_auc']:.4f}\n"
            report += f"  AUCPR:     {metrics['aucpr']:.4f}\n\n"
            
            # Precision@K
            report += "PRECISION@K:\n"
            for k in self.k_values:
                metric_key = f'precision_at_{k}'
                if metric_key in metrics:
                    report += f"  Precision@{k}: {metrics[metric_key]:.4f}\n"
            report += "\n"
            
            # Alert metrics
            report += "ALERT METRICS:\n"
            for threshold in [0.5, 0.7, 0.9]:
                alert_key = f'alert_rate_threshold_{threshold}'
                if alert_key in metrics:
                    report += f"  Alert Rate @{threshold}: {metrics[alert_key]:.4f}\n"
            report += "\n"
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        report += "CONFUSION MATRIX:\n"
        report += f"  True Negatives:  {cm[0][0]}\n"
        report += f"  False Positives: {cm[0][1]}\n"
        report += f"  False Negatives: {cm[1][0]}\n"
        report += f"  True Positives:  {cm[1][1]}\n\n"
        
        report += f"{'='*60}\n"
        
        return report
