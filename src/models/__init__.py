"""Base model class for phishing detection."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig


class BasePhishingDetector(ABC):
    """Abstract base class for phishing detection models."""
    
    def __init__(self, config: DictConfig):
        """Initialize the detector with configuration.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.feature_names = []
        self.is_trained = False
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        pass
    
    def predict_single(self, url: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Predict phishing risk for a single URL.
        
        Args:
            url: URL to analyze
            content: Optional web page content
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features (this would need feature extractor)
        # For now, return a placeholder
        features = np.random.random((1, 10))  # Placeholder
        
        proba = self.predict_proba(features)[0]
        prediction = self.predict(features)[0]
        
        return {
            'url': url,
            'is_phishing': bool(prediction),
            'phishing_probability': float(proba[1]),
            'legitimate_probability': float(proba[0]),
            'confidence': float(max(proba))
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores.
        
        Returns:
            Feature importance array or None if not available
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        import pickle
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'config': self.config
        }
