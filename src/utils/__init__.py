"""Utility functions for the phishing detection system."""

import hashlib
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device object
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def anonymize_pii(text: str) -> str:
    """Anonymize personally identifiable information in text.
    
    Args:
        text: Input text that may contain PII
        
    Returns:
        Text with PII replaced by placeholders
    """
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL_REDACTED]', text)
    
    # Phone numbers (various formats)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    
    # Credit card numbers (basic pattern)
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                  '[CARD_REDACTED]', text)
    
    # SSN (basic pattern)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
    
    return text


def hash_sensitive_data(data: str, salt: str = "phishing_detection") -> str:
    """Hash sensitive data for privacy protection.
    
    Args:
        data: Sensitive data to hash
        salt: Salt for hashing
        
    Returns:
        SHA-256 hash of the salted data
    """
    return hashlib.sha256((data + salt).encode()).hexdigest()[:16]


def validate_url(url: str) -> bool:
    """Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path
    """
    OmegaConf.save(config, output_path)


def time_function(func):
    """Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of text.
    
    Args:
        text: Input text
        
    Returns:
        Entropy value
    """
    if not text:
        return 0.0
    
    # Count character frequencies
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    text_length = len(text)
    for count in char_counts.values():
        probability = count / text_length
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def extract_domain_features(url: str) -> Dict[str, Any]:
    """Extract domain-related features from URL.
    
    Args:
        url: URL to analyze
        
    Returns:
        Dictionary of domain features
    """
    features = {}
    
    try:
        # Basic URL parsing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Extract domain (simplified)
        domain_match = re.search(r'https?://([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            features['domain_length'] = len(domain)
            features['num_subdomains'] = domain.count('.')
            features['has_www'] = domain.startswith('www.')
            
            # TLD analysis
            tld_match = re.search(r'\.([^.]+)$', domain)
            if tld_match:
                tld = tld_match.group(1).lower()
                suspicious_tlds = {'tk', 'ml', 'ga', 'cf', 'bit', 'onion'}
                features['suspicious_tld'] = tld in suspicious_tlds
            else:
                features['suspicious_tld'] = False
        else:
            features['domain_length'] = 0
            features['num_subdomains'] = 0
            features['has_www'] = False
            features['suspicious_tld'] = False
            
    except Exception:
        # Return default values if parsing fails
        features = {
            'domain_length': 0,
            'num_subdomains': 0,
            'has_www': False,
            'suspicious_tld': False
        }
    
    return features


def create_feature_names(feature_config: Dict[str, Any]) -> List[str]:
    """Create feature names based on configuration.
    
    Args:
        feature_config: Feature configuration dictionary
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    # URL features
    if feature_config.get('url', {}).get('length', False):
        feature_names.append('url_length')
    if feature_config.get('url', {}).get('num_dots', False):
        feature_names.append('num_dots')
    if feature_config.get('url', {}).get('has_https', False):
        feature_names.append('has_https')
    if feature_config.get('url', {}).get('num_special_chars', False):
        feature_names.append('num_special_chars')
    
    # Domain features
    if feature_config.get('domain', {}).get('age_days', False):
        feature_names.append('domain_age_days')
    if feature_config.get('domain', {}).get('suspicious_tld', False):
        feature_names.append('suspicious_tld')
    
    # Content features
    if feature_config.get('content', {}).get('text_length', False):
        feature_names.append('content_text_length')
    if feature_config.get('content', {}).get('suspicious_keywords', False):
        feature_names.append('suspicious_keywords_count')
    
    return feature_names
