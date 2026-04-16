"""Neural Network implementation for phishing detection."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models import BasePhishingDetector
from src.utils import get_device


class PhishingNeuralNetwork(nn.Module):
    """Neural network architecture for phishing detection."""
    
    def __init__(self, input_size: int, hidden_layers: list, dropout_rate: float = 0.3):
        """Initialize neural network.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class NeuralNetworkDetector(BasePhishingDetector):
    """Neural Network-based phishing detector."""
    
    def __init__(self, config: DictConfig):
        """Initialize Neural Network detector.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        self.device = get_device()
        self.scaler = StandardScaler()
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Training parameters
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.hidden_layers = config.hidden_layers
        self.dropout_rate = config.dropout_rate
        self.l2_regularization = config.l2_regularization
        
        # Early stopping
        self.early_stopping_patience = config.early_stopping.patience
        self.early_stopping_restore_best = config.early_stopping.restore_best_weights
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the Neural Network model.
        
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
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = PhishingNeuralNetwork(
            input_size=X.shape[1],
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.l2_regularization
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.early_stopping_restore_best:
                        best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    if self.early_stopping_restore_best:
                        self.model.load_state_dict(best_model_state)
                    break
        
        self.is_trained = True
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Classification report
        report = classification_report(y, predictions, output_dict=True)
        
        # Precision-Recall curve metrics
        precision, recall, thresholds = precision_recall_curve(y, probabilities[:, 1])
        aucpr = np.trapz(precision, recall)
        
        metrics = {
            'training_accuracy': report['accuracy'],
            'training_precision': report['1']['precision'],
            'training_recall': report['1']['recall'],
            'training_f1': report['1']['f1-score'],
            'aucpr': aucpr,
            'epochs_trained': epoch + 1,
            'final_loss': epoch_loss / len(dataloader)
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_probabilities = self.predict_proba(X_val)
            
            val_report = classification_report(y_val, val_predictions, output_dict=True)
            val_precision, val_recall, _ = precision_recall_curve(y_val, val_probabilities[:, 1])
            val_aucpr = np.trapz(val_precision, val_recall)
            
            metrics.update({
                'validation_accuracy': val_report['accuracy'],
                'validation_precision': val_report['1']['precision'],
                'validation_recall': val_report['1']['recall'],
                'validation_f1': val_report['1']['f1-score'],
                'validation_aucpr': val_aucpr
            })
        
        return metrics
    
    def _validate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Validate the model.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Validation loss
        """
        self.model.eval()
        
        X_val_scaled = self.scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            loss = self.criterion(outputs, y_val_tensor)
        
        return loss.item()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores (not directly available for neural networks).
        
        Returns:
            None (feature importance not directly available)
        """
        # Neural networks don't have direct feature importance
        # Would need to use techniques like SHAP or permutation importance
        return None
    
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
