"""Random Forest implementation for phishing detection."""

from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve

from src.models import BasePhishingDetector


class RandomForestDetector(BasePhishingDetector):
    """Random Forest-based phishing detector."""
    
    def __init__(self, config: DictConfig):
        """Initialize Random Forest detector.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Initialize Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            bootstrap=config.bootstrap,
            random_state=config.random_state,
            n_jobs=config.n_jobs
        )
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        # Set feature names if not already set
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Precision-Recall curve metrics
        precision, recall, thresholds = precision_recall_curve(y, y_proba[:, 1])
        
        # Calculate AUCPR
        aucpr = np.trapz(precision, recall)
        
        metrics = {
            'training_accuracy': report['accuracy'],
            'training_precision': report['1']['precision'],
            'training_recall': report['1']['recall'],
            'training_f1': report['1']['f1-score'],
            'aucpr': aucpr,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'feature_importance': self.get_feature_importance()
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            y_val_proba = self.predict_proba(X_val)
            
            val_report = classification_report(y_val, y_val_pred, output_dict=True)
            val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_proba[:, 1])
            val_aucpr = np.trapz(val_precision, val_recall)
            
            metrics.update({
                'validation_accuracy': val_report['accuracy'],
                'validation_precision': val_report['1']['precision'],
                'validation_recall': val_report['1']['recall'],
                'validation_f1': val_report['1']['f1-score'],
                'validation_aucpr': val_aucpr
            })
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            return None
        
        return self.model.feature_importances_
    
    def get_feature_importance_dict(self) -> Dict[str, float]:
        """Get feature importance as dictionary.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = self.get_feature_importance()
        if importance is None:
            return {}
        
        return dict(zip(self.feature_names, importance))
    
    def get_top_features(self, n: int = 10) -> Dict[str, float]:
        """Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary of top features and their importance scores
        """
        importance_dict = self.get_feature_importance_dict()
        
        # Sort by importance (descending)
        sorted_features = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return dict(sorted_features[:n])
    
    def predict_with_confidence(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions with confidence scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Confidence is the maximum probability
        confidence = np.max(probabilities, axis=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence
        }
