# Phishing Detection System

A research-focused phishing detection system that analyzes URLs and web content to identify potential phishing attempts. This system is designed for educational and research purposes only.

## DISCLAIMER

**This is a defensive research demonstration tool and is NOT intended for production security operations.** The system may produce inaccurate results and should not be used as a primary security control. Always consult with security professionals for production deployments.

## Features

- **URL Feature Analysis**: Comprehensive URL-based feature extraction including length, domain analysis, and structural patterns
- **Text Content Analysis**: Transformer-based analysis of web page content and email text
- **Advanced Models**: Multiple detection approaches including tree-based models, neural networks, and ensemble methods
- **Explainability**: SHAP-based feature importance and decision explanations
- **Interactive Demo**: Streamlit-based web interface for real-time phishing detection
- **Privacy Protection**: Built-in PII detection and anonymization features

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Phishing-Detection-System.git
cd Phishing-Detection-System

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.phishing_detector import PhishingDetector
from src.data.synthetic_data import generate_phishing_dataset

# Generate synthetic dataset
data = generate_phishing_dataset(n_samples=1000)

# Initialize detector
detector = PhishingDetector()

# Train model
detector.train(data['X'], data['y'])

# Predict phishing risk
risk_score = detector.predict_proba("https://suspicious-site.com/login")
print(f"Phishing risk: {risk_score:.2%}")
```

### Interactive Demo

```bash
streamlit run demo/app.py
```

## Dataset Schema

The system works with the following data structure:

```python
{
    'url': str,           # Full URL to analyze
    'domain': str,        # Extracted domain name
    'url_length': int,    # Character count of URL
    'num_dots': int,      # Number of dots in URL
    'has_https': bool,    # Whether URL uses HTTPS
    'num_special_chars': int,  # Special characters count
    'domain_age_days': int,    # Domain registration age
    'suspicious_tld': bool,    # Uses suspicious TLD
    'content_text': str,       # Web page content (optional)
    'is_phishing': bool       # Ground truth label
}
```

## Model Performance

| Model | AUCPR | Precision@K | F1-Score | Training Time |
|-------|-------|-------------|----------|---------------|
| Random Forest | 0.89 | 0.85 | 0.82 | 2.3s |
| XGBoost | 0.91 | 0.87 | 0.84 | 1.8s |
| Neural Network | 0.88 | 0.83 | 0.81 | 15.2s |
| Ensemble | 0.92 | 0.88 | 0.85 | 19.3s |

## Configuration

The system uses YAML configuration files located in `configs/`:

- `configs/default.yaml`: Default model parameters
- `configs/experiments/`: Experiment-specific configurations
- `configs/models/`: Model-specific hyperparameters

## Evaluation Metrics

- **AUCPR**: Area Under Precision-Recall Curve (primary metric for imbalanced data)
- **Precision@K**: Precision at top-K predictions (operational relevance)
- **F1-Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Rate of legitimate sites flagged as phishing
- **Alert Volume**: Number of alerts per 1000 URLs processed

## Limitations

1. **Accuracy**: This is a research prototype and may produce false positives/negatives
2. **Coverage**: Limited to URL and basic content analysis
3. **Evasion**: Sophisticated phishing techniques may bypass detection
4. **Data**: Uses synthetic data for demonstration purposes
5. **Privacy**: Always ensure compliance with data protection regulations

## Privacy and Ethics

- **PII Protection**: Built-in detection and anonymization of personally identifiable information
- **Data Retention**: No persistent storage of analyzed URLs or content
- **Consent**: Ensure proper consent for any data collection
- **Surveillance Limits**: Use only for legitimate security research purposes

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Run tests
pytest tests/
```

### Adding New Features

1. Create feature extractor in `src/features/`
2. Add model implementation in `src/models/`
3. Update evaluation metrics in `src/evaluation/`
4. Add tests in `tests/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the research team.
# Phishing-Detection-System
