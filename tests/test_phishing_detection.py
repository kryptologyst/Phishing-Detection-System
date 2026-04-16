"""Basic tests for the phishing detection system."""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data import PhishingDataGenerator
from src.features import PhishingFeatureExtractor
from src.models.random_forest import RandomForestDetector
from src.eval import PhishingEvaluator
from src.utils import load_config


class TestPhishingDetection:
    """Test cases for phishing detection system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = load_config("configs/default.yaml")
        self.data_generator = PhishingDataGenerator(self.config)
        self.feature_extractor = PhishingFeatureExtractor(self.config)
        self.evaluator = PhishingEvaluator()
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        dataset = self.data_generator.generate_dataset(n_samples=100)
        
        assert 'X' in dataset
        assert 'y' in dataset
        assert 'feature_names' in dataset
        assert 'metadata' in dataset
        
        assert dataset['X'].shape[0] == 100
        assert dataset['y'].shape[0] == 100
        assert len(dataset['feature_names']) > 0
        
        # Check that we have both phishing and legitimate samples
        assert np.sum(dataset['y']) > 0  # At least some phishing samples
        assert np.sum(dataset['y']) < len(dataset['y'])  # Not all phishing
    
    def test_feature_extraction(self):
        """Test feature extraction from URLs."""
        test_urls = [
            "https://google.com",
            "https://secure-bank-login.tk/verify",
            "https://facebook.com/login"
        ]
        
        for url in test_urls:
            features = self.feature_extractor.extract_url_features(url)
            
            assert isinstance(features, dict)
            assert 'url_length' in features
            assert 'num_dots' in features
            assert 'has_https' in features
            assert 'domain_length' in features
            
            # Check that features are numeric
            for key, value in features.items():
                assert isinstance(value, (int, float))
    
    def test_random_forest_model(self):
        """Test Random Forest model training and prediction."""
        # Generate small dataset
        dataset = self.data_generator.generate_dataset(n_samples=200)
        X, y = dataset['X'], dataset['y']
        
        # Create train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Initialize and train model
        model = RandomForestDetector(self.config)
        metrics = model.train(X_train, y_train, X_test, y_test)
        
        # Check training metrics
        assert 'training_accuracy' in metrics
        assert 'aucpr' in metrics
        assert metrics['training_accuracy'] > 0
        assert metrics['aucpr'] > 0
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(y_test)
        assert probabilities.shape[0] == len(y_test)
        assert probabilities.shape[1] == 2  # Binary classification
        
        # Check that predictions are valid
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob in probabilities.flatten())
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        # Generate test data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        y_proba = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4],
            [0.4, 0.6], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]
        ])
        
        metrics = self.evaluator.evaluate_model(y_true, y_pred, y_proba)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'aucpr']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1
    
    def test_url_validation(self):
        """Test URL validation utility."""
        from src.utils import validate_url
        
        valid_urls = [
            "https://google.com",
            "http://example.com",
            "https://subdomain.example.com/path"
        ]
        
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "javascript:alert('test')"
        ]
        
        for url in valid_urls:
            assert validate_url(url), f"URL should be valid: {url}"
        
        for url in invalid_urls:
            assert not validate_url(url), f"URL should be invalid: {url}"
    
    def test_pii_anonymization(self):
        """Test PII anonymization."""
        from src.utils import anonymize_pii
        
        text_with_pii = "Contact us at john@example.com or call 555-123-4567"
        anonymized = anonymize_pii(text_with_pii)
        
        assert "[EMAIL_REDACTED]" in anonymized
        assert "[PHONE_REDACTED]" in anonymized
        assert "john@example.com" not in anonymized
        assert "555-123-4567" not in anonymized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
