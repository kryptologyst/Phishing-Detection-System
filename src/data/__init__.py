"""Data generation and processing utilities for phishing detection."""

import hashlib
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils import anonymize_pii, hash_sensitive_data, validate_url


class PhishingDataGenerator:
    """Generate synthetic phishing detection datasets."""
    
    def __init__(self, config: DictConfig):
        """Initialize data generator with configuration.
        
        Args:
            config: Configuration object containing data generation parameters
        """
        self.config = config
        self.random_state = config.data.synthetic.random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Phishing patterns
        self.phishing_patterns = {
            'suspicious_domains': [
                'secure-bank-login', 'paypal-security', 'amazon-account',
                'microsoft-support', 'apple-id-verify', 'google-security',
                'facebook-login', 'twitter-verify', 'instagram-security'
            ],
            'suspicious_tlds': ['tk', 'ml', 'ga', 'cf', 'bit', 'onion'],
            'suspicious_keywords': [
                'urgent', 'verify', 'suspended', 'expired', 'security',
                'update', 'confirm', 'validate', 'restore', 'unlock'
            ],
            'legitimate_domains': [
                'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
                'apple.com', 'paypal.com', 'twitter.com', 'instagram.com',
                'linkedin.com', 'github.com', 'stackoverflow.com'
            ]
        }
    
    def generate_url_features(self, is_phishing: bool) -> Dict[str, Any]:
        """Generate URL-based features for a sample.
        
        Args:
            is_phishing: Whether this is a phishing URL
            
        Returns:
            Dictionary of URL features
        """
        features = {}
        
        if is_phishing:
            # Generate phishing-like features
            features['url_length'] = np.random.randint(50, 200)
            features['num_dots'] = np.random.randint(3, 8)
            features['has_https'] = np.random.choice([0, 1], p=[0.6, 0.4])
            features['num_special_chars'] = np.random.randint(5, 15)
            features['has_www'] = np.random.choice([0, 1], p=[0.7, 0.3])
            features['path_depth'] = np.random.randint(2, 6)
            features['query_length'] = np.random.randint(20, 100)
            features['domain_age_days'] = np.random.randint(1, 30)
            features['suspicious_tld'] = np.random.choice([0, 1], p=[0.3, 0.7])
            features['num_hyphens'] = np.random.randint(2, 8)
            features['num_underscores'] = np.random.randint(1, 5)
        else:
            # Generate legitimate-like features
            features['url_length'] = np.random.randint(20, 80)
            features['num_dots'] = np.random.randint(1, 3)
            features['has_https'] = np.random.choice([0, 1], p=[0.2, 0.8])
            features['num_special_chars'] = np.random.randint(1, 5)
            features['has_www'] = np.random.choice([0, 1], p=[0.4, 0.6])
            features['path_depth'] = np.random.randint(1, 3)
            features['query_length'] = np.random.randint(5, 30)
            features['domain_age_days'] = np.random.randint(365, 3650)
            features['suspicious_tld'] = 0
            features['num_hyphens'] = np.random.randint(0, 2)
            features['num_underscores'] = np.random.randint(0, 1)
        
        return features
    
    def generate_content_features(self, is_phishing: bool) -> Dict[str, Any]:
        """Generate content-based features for a sample.
        
        Args:
            is_phishing: Whether this is phishing content
            
        Returns:
            Dictionary of content features
        """
        features = {}
        
        if is_phishing:
            # Generate phishing-like content features
            features['content_text_length'] = np.random.randint(500, 2000)
            features['suspicious_keywords_count'] = np.random.randint(3, 8)
            features['form_count'] = np.random.randint(2, 5)
            features['external_links'] = np.random.randint(5, 15)
            features['entropy'] = np.random.uniform(4.5, 5.5)
        else:
            # Generate legitimate-like content features
            features['content_text_length'] = np.random.randint(1000, 5000)
            features['suspicious_keywords_count'] = np.random.randint(0, 2)
            features['form_count'] = np.random.randint(0, 2)
            features['external_links'] = np.random.randint(10, 50)
            features['entropy'] = np.random.uniform(4.0, 4.8)
        
        return features
    
    def generate_url_string(self, is_phishing: bool) -> str:
        """Generate a realistic URL string.
        
        Args:
            is_phishing: Whether this is a phishing URL
            
        Returns:
            Generated URL string
        """
        if is_phishing:
            # Generate phishing URL
            domain = random.choice(self.phishing_patterns['suspicious_domains'])
            tld = random.choice(self.phishing_patterns['suspicious_tlds'])
            protocol = random.choice(['http://', 'https://'])
            path = '/'.join([f"page{i}" for i in range(random.randint(1, 3))])
            query = f"?id={random.randint(1000, 9999)}&token={random.randint(10000, 99999)}"
            
            return f"{protocol}{domain}.{tld}/{path}{query}"
        else:
            # Generate legitimate URL
            domain = random.choice(self.phishing_patterns['legitimate_domains'])
            protocol = random.choice(['http://', 'https://'])
            path = '/'.join([f"section{i}" for i in range(random.randint(0, 2))])
            query = f"?ref={random.randint(100, 999)}" if random.random() > 0.5 else ""
            
            return f"{protocol}{domain}/{path}{query}"
    
    def generate_dataset(self, n_samples: Optional[int] = None) -> Dict[str, Any]:
        """Generate synthetic phishing detection dataset.
        
        Args:
            n_samples: Number of samples to generate (uses config default if None)
            
        Returns:
            Dictionary containing features, labels, and metadata
        """
        if n_samples is None:
            n_samples = self.config.data.synthetic.n_samples
        
        # Generate labels (imbalanced dataset)
        phishing_ratio = 0.3  # 30% phishing, 70% legitimate
        n_phishing = int(n_samples * phishing_ratio)
        n_legitimate = n_samples - n_phishing
        
        labels = [1] * n_phishing + [0] * n_legitimate
        random.shuffle(labels)
        
        # Generate features
        urls = []
        url_features = []
        content_features = []
        
        for is_phishing in labels:
            # Generate URL string
            url = self.generate_url_string(bool(is_phishing))
            urls.append(url)
            
            # Generate features
            url_feats = self.generate_url_features(bool(is_phishing))
            content_feats = self.generate_content_features(bool(is_phishing))
            
            url_features.append(url_feats)
            content_features.append(content_feats)
        
        # Combine features
        all_features = []
        feature_names = []
        
        for i in range(len(url_features)):
            combined = {**url_features[i], **content_features[i]}
            all_features.append(list(combined.values()))
            
            if not feature_names:  # Set feature names once
                feature_names = list(combined.keys())
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(labels)
        
        return {
            'X': X,
            'y': y,
            'urls': urls,
            'feature_names': feature_names,
            'metadata': {
                'n_samples': n_samples,
                'n_phishing': n_phishing,
                'n_legitimate': n_legitimate,
                'phishing_ratio': phishing_ratio,
                'generated_at': datetime.now().isoformat()
            }
        }


class DataProcessor:
    """Process and clean phishing detection data."""
    
    def __init__(self, config: DictConfig):
        """Initialize data processor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_names = []
    
    def preprocess_urls(self, urls: List[str]) -> List[str]:
        """Preprocess URLs for analysis.
        
        Args:
            urls: List of URLs to preprocess
            
        Returns:
            List of preprocessed URLs
        """
        processed_urls = []
        
        for url in urls:
            # Validate URL
            if not validate_url(url):
                continue
            
            # Anonymize PII if configured
            if self.config.data.preprocessing.anonymize_pii:
                url = anonymize_pii(url)
            
            # Hash IPs if configured
            if self.config.data.preprocessing.hash_ips:
                # Simple IP detection and hashing
                ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
                if re.search(ip_pattern, url):
                    url = re.sub(ip_pattern, lambda m: hash_sensitive_data(m.group()), url)
            
            processed_urls.append(url)
        
        return processed_urls
    
    def extract_features(self, urls: List[str]) -> np.ndarray:
        """Extract features from URLs.
        
        Args:
            urls: List of URLs to extract features from
            
        Returns:
            Feature matrix
        """
        features = []
        
        for url in urls:
            url_features = self._extract_url_features(url)
            content_features = self._extract_content_features(url)
            
            combined_features = {**url_features, **content_features}
            features.append(list(combined_features.values()))
            
            if not self.feature_names:
                self.feature_names = list(combined_features.keys())
        
        return np.array(features)
    
    def _extract_url_features(self, url: str) -> Dict[str, Any]:
        """Extract URL-based features.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary of URL features
        """
        features = {}
        
        # Basic URL features
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9.-]', url))
        features['has_https'] = 1 if url.startswith('https://') else 0
        features['has_www'] = 1 if 'www.' in url else 0
        
        # Path analysis
        try:
            path_start = url.find('/', 8)  # After protocol
            if path_start != -1:
                path = url[path_start:]
                features['path_depth'] = path.count('/')
                features['query_length'] = len(url.split('?')[1]) if '?' in url else 0
            else:
                features['path_depth'] = 0
                features['query_length'] = 0
        except Exception:
            features['path_depth'] = 0
            features['query_length'] = 0
        
        # Domain features
        try:
            domain_match = re.search(r'https?://([^/]+)', url)
            if domain_match:
                domain = domain_match.group(1)
                features['domain_length'] = len(domain)
                features['num_subdomains'] = domain.count('.')
                
                # TLD analysis
                tld_match = re.search(r'\.([^.]+)$', domain)
                if tld_match:
                    tld = tld_match.group(1).lower()
                    suspicious_tlds = {'tk', 'ml', 'ga', 'cf', 'bit', 'onion'}
                    features['suspicious_tld'] = 1 if tld in suspicious_tlds else 0
                else:
                    features['suspicious_tld'] = 0
            else:
                features['domain_length'] = 0
                features['num_subdomains'] = 0
                features['suspicious_tld'] = 0
        except Exception:
            features['domain_length'] = 0
            features['num_subdomains'] = 0
            features['suspicious_tld'] = 0
        
        # Simulate domain age (in real implementation, would query WHOIS)
        features['domain_age_days'] = np.random.randint(1, 3650)
        
        return features
    
    def _extract_content_features(self, url: str) -> Dict[str, Any]:
        """Extract content-based features (simplified).
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary of content features
        """
        features = {}
        
        # Simulate content analysis (in real implementation, would fetch page)
        features['content_text_length'] = np.random.randint(100, 5000)
        features['suspicious_keywords_count'] = np.random.randint(0, 10)
        features['form_count'] = np.random.randint(0, 5)
        features['external_links'] = np.random.randint(0, 20)
        
        # Calculate entropy
        features['entropy'] = self._calculate_entropy(url)
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text.
        
        Args:
            text: Input text
            
        Returns:
            Entropy value
        """
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        text_length = len(text)
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def create_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                               test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """Create train-test split with stratification.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
