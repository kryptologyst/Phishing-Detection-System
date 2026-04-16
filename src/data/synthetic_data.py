"""Synthetic data generation for phishing detection."""

from src.data import PhishingDataGenerator
from src.utils import load_config


def generate_phishing_dataset(n_samples: int = 10000, 
                            config_path: str = "configs/default.yaml") -> dict:
    """Generate synthetic phishing detection dataset.
    
    Args:
        n_samples: Number of samples to generate
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing dataset and metadata
    """
    config = load_config(config_path)
    generator = PhishingDataGenerator(config)
    
    return generator.generate_dataset(n_samples)


if __name__ == "__main__":
    # Generate dataset for testing
    dataset = generate_phishing_dataset(n_samples=1000)
    
    print(f"Generated dataset with {dataset['metadata']['n_samples']} samples")
    print(f"Phishing ratio: {dataset['metadata']['phishing_ratio']:.2%}")
    print(f"Feature names: {dataset['feature_names']}")
    print(f"Feature matrix shape: {dataset['X'].shape}")
    print(f"Labels shape: {dataset['y'].shape}")
