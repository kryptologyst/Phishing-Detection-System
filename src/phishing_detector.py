"""Main phishing detection system."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data import PhishingDataGenerator, DataProcessor
from src.features import PhishingFeatureExtractor
from src.models import BasePhishingDetector
from src.models.random_forest import RandomForestDetector
from src.models.xgboost_detector import XGBoostDetector
from src.models.neural_network import NeuralNetworkDetector
from src.eval import PhishingEvaluator
from src.utils import load_config, set_deterministic_seed, setup_logging


class PhishingDetector:
    """Main phishing detection system that orchestrates all components."""
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize the phishing detection system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Set up logging
        self.logger = setup_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.file
        )
        
        # Set deterministic seed
        set_deterministic_seed(self.config.data.synthetic.random_state)
        
        # Initialize components
        self.data_generator = PhishingDataGenerator(self.config)
        self.data_processor = DataProcessor(self.config)
        self.feature_extractor = PhishingFeatureExtractor(self.config)
        self.evaluator = PhishingEvaluator(
            k_values=self.config.evaluation.k_values,
            target_precision=self.config.evaluation.target_precision
        )
        
        # Model registry
        self.models = {}
        self.trained_models = {}
        
        self.logger.info("Phishing detection system initialized")
    
    def register_model(self, name: str, model: BasePhishingDetector) -> None:
        """Register a model for training and evaluation.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.models[name] = model
        self.logger.info(f"Registered model: {name}")
    
    def load_default_models(self) -> None:
        """Load default models based on configuration."""
        try:
            # Random Forest
            rf_config = OmegaConf.load("configs/models/random_forest.yaml")
            self.register_model("random_forest", RandomForestDetector(rf_config))
            
            # XGBoost (if available)
            try:
                xgb_config = OmegaConf.load("configs/models/xgboost.yaml")
                self.register_model("xgboost", XGBoostDetector(xgb_config))
            except ImportError:
                self.logger.warning("XGBoost not available, skipping XGBoost model")
            
            # Neural Network
            nn_config = OmegaConf.load("configs/models/neural_network.yaml")
            self.register_model("neural_network", NeuralNetworkDetector(nn_config))
            
        except Exception as e:
            self.logger.error(f"Error loading default models: {e}")
    
    def generate_dataset(self, n_samples: Optional[int] = None) -> Dict[str, Any]:
        """Generate synthetic phishing dataset.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated dataset
        """
        self.logger.info("Generating synthetic dataset...")
        dataset = self.data_generator.generate_dataset(n_samples)
        self.logger.info(f"Generated dataset with {dataset['metadata']['n_samples']} samples")
        return dataset
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                   X_val: Optional[np.ndarray] = None, 
                   y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        self.logger.info(f"Training model: {model_name}")
        
        model = self.models[model_name]
        metrics = model.train(X, y, X_val, y_val)
        
        # Store trained model
        self.trained_models[model_name] = model
        
        self.logger.info(f"Model {model_name} trained successfully")
        return metrics
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """Train all registered models.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics for all models
        """
        all_metrics = {}
        
        for model_name in self.models:
            try:
                metrics = self.train_model(model_name, X, y, X_val, y_val)
                all_metrics[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error training model {model_name}: {e}")
                all_metrics[model_name] = {"error": str(e)}
        
        return all_metrics
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate a trained model.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.trained_models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Evaluate
        metrics = self.evaluator.evaluate_model(y_test, y_pred, y_proba)
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics for all models
        """
        all_metrics = {}
        
        for model_name in self.trained_models:
            try:
                metrics = self.evaluate_model(model_name, X_test, y_test)
                all_metrics[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {e}")
                all_metrics[model_name] = {"error": str(e)}
        
        return all_metrics
    
    def create_leaderboard(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create a leaderboard from evaluation results.
        
        Args:
            evaluation_results: Evaluation results for all models
            
        Returns:
            Leaderboard with ranked models
        """
        return self.evaluator.create_leaderboard(evaluation_results)
    
    def predict_phishing_risk(self, url: str, model_name: str = "random_forest",
                             content: Optional[str] = None) -> Dict[str, Any]:
        """Predict phishing risk for a single URL.
        
        Args:
            url: URL to analyze
            model_name: Name of the model to use
            content: Optional web page content
            
        Returns:
            Prediction results
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.trained_models[model_name]
        
        # Extract features
        features = self.feature_extractor.extract_all_features(url, content)
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        return {
            'url': url,
            'is_phishing': bool(prediction),
            'phishing_probability': float(probabilities[1]),
            'legitimate_probability': float(probabilities[0]),
            'confidence': float(max(probabilities)),
            'model_used': model_name,
            'features_extracted': len(features)
        }
    
    def batch_predict(self, urls: List[str], model_name: str = "random_forest",
                     contents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Predict phishing risk for multiple URLs.
        
        Args:
            urls: List of URLs to analyze
            model_name: Name of the model to use
            contents: Optional list of web page contents
            
        Returns:
            List of prediction results
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        if contents is None:
            contents = [None] * len(urls)
        
        results = []
        for url, content in zip(urls, contents):
            try:
                result = self.predict_phishing_risk(url, model_name, content)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error predicting for URL {url}: {e}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'is_phishing': False,
                    'phishing_probability': 0.0
                })
        
        return results
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get feature importance for a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of feature importance or None
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'get_feature_importance_dict'):
            return model.get_feature_importance_dict()
        elif hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            if importance is not None:
                return dict(zip(self.feature_extractor.get_feature_names(), importance))
        
        return None
    
    def run_full_pipeline(self, n_samples: int = 10000) -> Dict[str, Any]:
        """Run the complete phishing detection pipeline.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Complete pipeline results
        """
        self.logger.info("Starting full phishing detection pipeline...")
        
        # Generate dataset
        dataset = self.generate_dataset(n_samples)
        X, y = dataset['X'], dataset['y']
        
        # Create train-test split
        X_train, X_test, y_train, y_test = self.data_processor.create_train_test_split(
            X, y, 
            test_size=self.config.data.synthetic.test_size,
            random_state=self.config.data.synthetic.random_state
        )
        
        # Load default models
        self.load_default_models()
        
        # Train all models
        self.logger.info("Training all models...")
        training_metrics = self.train_all_models(X_train, y_train, X_test, y_test)
        
        # Evaluate all models
        self.logger.info("Evaluating all models...")
        evaluation_metrics = self.evaluate_all_models(X_test, y_test)
        
        # Create leaderboard
        leaderboard = self.create_leaderboard(evaluation_metrics)
        
        # Compile results
        results = {
            'dataset_info': dataset['metadata'],
            'training_metrics': training_metrics,
            'evaluation_metrics': evaluation_metrics,
            'leaderboard': leaderboard,
            'feature_names': dataset['feature_names'],
            'models_trained': list(self.trained_models.keys())
        }
        
        self.logger.info("Full pipeline completed successfully")
        return results
