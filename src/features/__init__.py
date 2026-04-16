"""Feature extraction for phishing detection."""

import re
from typing import Any, Dict, List, Optional

import numpy as np
import tldextract
from omegaconf import DictConfig

from src.utils import calculate_entropy, extract_domain_features


class PhishingFeatureExtractor:
    """Extract features from URLs and content for phishing detection."""
    
    def __init__(self, config: DictConfig):
        """Initialize feature extractor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_names = []
        
        # Suspicious patterns
        self.suspicious_keywords = [
            'urgent', 'verify', 'suspended', 'expired', 'security',
            'update', 'confirm', 'validate', 'restore', 'unlock',
            'account', 'login', 'password', 'credit', 'card',
            'bank', 'paypal', 'amazon', 'microsoft', 'apple'
        ]
        
        self.suspicious_tlds = {'tk', 'ml', 'ga', 'cf', 'bit', 'onion'}
        
    def extract_url_features(self, url: str) -> Dict[str, Any]:
        """Extract URL-based features.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary of URL features
        """
        features = {}
        
        # Basic URL structure features
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9.-]', url))
        features['has_https'] = 1 if url.startswith('https://') else 0
        features['has_www'] = 1 if 'www.' in url else 0
        
        # Path analysis
        features.update(self._extract_path_features(url))
        
        # Domain analysis
        features.update(self._extract_domain_features(url))
        
        # Entropy and randomness
        features['url_entropy'] = calculate_entropy(url)
        features['has_random_string'] = self._has_random_string(url)
        
        return features
    
    def extract_content_features(self, content: Optional[str] = None) -> Dict[str, Any]:
        """Extract content-based features.
        
        Args:
            content: Web page content (optional)
            
        Returns:
            Dictionary of content features
        """
        features = {}
        
        if content is None:
            # Return default values if no content
            features.update({
                'content_text_length': 0,
                'suspicious_keywords_count': 0,
                'form_count': 0,
                'external_links': 0,
                'content_entropy': 0.0,
                'has_login_form': 0,
                'has_password_field': 0,
                'has_credit_card_field': 0
            })
            return features
        
        # Text analysis
        features['content_text_length'] = len(content)
        features['content_entropy'] = calculate_entropy(content)
        
        # Suspicious keyword detection
        content_lower = content.lower()
        features['suspicious_keywords_count'] = sum(
            1 for keyword in self.suspicious_keywords 
            if keyword in content_lower
        )
        
        # Form analysis
        features['form_count'] = content.count('<form')
        features['has_login_form'] = 1 if 'login' in content_lower and '<form' in content else 0
        features['has_password_field'] = 1 if 'password' in content_lower else 0
        features['has_credit_card_field'] = 1 if any(
            term in content_lower for term in ['credit', 'card', 'cvv', 'cvc']
        ) else 0
        
        # Link analysis
        features['external_links'] = len(re.findall(r'href=["\']http', content))
        
        return features
    
    def extract_all_features(self, url: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Extract all features from URL and content.
        
        Args:
            url: URL to analyze
            content: Optional web page content
            
        Returns:
            Dictionary of all features
        """
        url_features = self.extract_url_features(url)
        content_features = self.extract_content_features(content)
        
        # Combine features
        all_features = {**url_features, **content_features}
        
        # Set feature names if not already set
        if not self.feature_names:
            self.feature_names = list(all_features.keys())
        
        return all_features
    
    def _extract_path_features(self, url: str) -> Dict[str, Any]:
        """Extract path-related features.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary of path features
        """
        features = {}
        
        try:
            # Find path start (after protocol and domain)
            path_start = url.find('/', 8)  # After http:// or https://
            if path_start != -1:
                path = url[path_start:]
                
                # Path depth
                features['path_depth'] = path.count('/')
                
                # Query parameters
                if '?' in path:
                    query_part = path.split('?')[1]
                    features['query_length'] = len(query_part)
                    features['num_query_params'] = query_part.count('&') + 1
                else:
                    features['query_length'] = 0
                    features['num_query_params'] = 0
                
                # File extension
                if '.' in path and '/' in path.split('.')[-1]:
                    features['has_file_extension'] = 1
                else:
                    features['has_file_extension'] = 0
                    
            else:
                features['path_depth'] = 0
                features['query_length'] = 0
                features['num_query_params'] = 0
                features['has_file_extension'] = 0
                
        except Exception:
            features['path_depth'] = 0
            features['query_length'] = 0
            features['num_query_params'] = 0
            features['has_file_extension'] = 0
        
        return features
    
    def _extract_domain_features(self, url: str) -> Dict[str, Any]:
        """Extract domain-related features.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary of domain features
        """
        features = {}
        
        try:
            # Use tldextract for better domain parsing
            extracted = tldextract.extract(url)
            
            if extracted.domain:
                domain = f"{extracted.domain}.{extracted.suffix}"
                features['domain_length'] = len(domain)
                features['domain_entropy'] = calculate_entropy(domain)
                
                # Subdomain analysis
                if extracted.subdomain:
                    features['num_subdomains'] = extracted.subdomain.count('.') + 1
                    features['subdomain_length'] = len(extracted.subdomain)
                else:
                    features['num_subdomains'] = 0
                    features['subdomain_length'] = 0
                
                # TLD analysis
                features['suspicious_tld'] = 1 if extracted.suffix.lower() in self.suspicious_tlds else 0
                features['tld_length'] = len(extracted.suffix)
                
                # Domain characteristics
                features['has_numbers_in_domain'] = 1 if re.search(r'\d', extracted.domain) else 0
                features['has_hyphens_in_domain'] = 1 if '-' in extracted.domain else 0
                
            else:
                # Default values if domain extraction fails
                features.update({
                    'domain_length': 0,
                    'domain_entropy': 0.0,
                    'num_subdomains': 0,
                    'subdomain_length': 0,
                    'suspicious_tld': 0,
                    'tld_length': 0,
                    'has_numbers_in_domain': 0,
                    'has_hyphens_in_domain': 0
                })
                
        except Exception:
            # Default values if parsing fails
            features.update({
                'domain_length': 0,
                'domain_entropy': 0.0,
                'num_subdomains': 0,
                'subdomain_length': 0,
                'suspicious_tld': 0,
                'tld_length': 0,
                'has_numbers_in_domain': 0,
                'has_hyphens_in_domain': 0
            })
        
        # Simulate domain age (in real implementation, would query WHOIS)
        features['domain_age_days'] = np.random.randint(1, 3650)
        
        return features
    
    def _has_random_string(self, url: str) -> int:
        """Check if URL contains random-looking strings.
        
        Args:
            url: URL to analyze
            
        Returns:
            1 if random string detected, 0 otherwise
        """
        # Look for patterns that suggest random generation
        random_patterns = [
            r'[a-z]{8,}',  # Long lowercase strings
            r'[A-Z]{8,}',  # Long uppercase strings
            r'\d{6,}',     # Long number sequences
            r'[a-zA-Z0-9]{10,}',  # Long alphanumeric strings
        ]
        
        for pattern in random_patterns:
            if re.search(pattern, url):
                return 1
        
        return 0
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
    
    def create_feature_matrix(self, urls: List[str], 
                            contents: Optional[List[str]] = None) -> np.ndarray:
        """Create feature matrix from URLs and contents.
        
        Args:
            urls: List of URLs to analyze
            contents: Optional list of web page contents
            
        Returns:
            Feature matrix
        """
        if contents is None:
            contents = [None] * len(urls)
        
        features = []
        for url, content in zip(urls, contents):
            feature_dict = self.extract_all_features(url, content)
            features.append(list(feature_dict.values()))
            
            if not self.feature_names:
                self.feature_names = list(feature_dict.keys())
        
        return np.array(features)
