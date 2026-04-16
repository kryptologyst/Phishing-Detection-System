"""Simple test script for the phishing detection system."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.phishing_detector import PhishingDetector
from src.data.synthetic_data import generate_phishing_dataset


def test_basic_functionality():
    """Test basic functionality of the phishing detection system."""
    print("Testing Phishing Detection System...")
    
    try:
        # Initialize detector
        print("1. Initializing detector...")
        detector = PhishingDetector()
        
        # Generate small dataset
        print("2. Generating test dataset...")
        dataset = detector.generate_dataset(n_samples=1000)
        print(f"   Generated {dataset['metadata']['n_samples']} samples")
        
        # Create train-test split
        print("3. Creating train-test split...")
        X, y = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = detector.data_processor.create_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Load and train Random Forest model
        print("4. Training Random Forest model...")
        detector.load_default_models()
        
        if 'random_forest' in detector.models:
            training_metrics = detector.train_model('random_forest', X_train, y_train, X_test, y_test)
            print(f"   Training accuracy: {training_metrics['training_accuracy']:.3f}")
            print(f"   AUCPR: {training_metrics['aucpr']:.3f}")
        
        # Test single prediction
        print("5. Testing single URL prediction...")
        test_url = "https://secure-bank-login.tk/verify-account"
        result = detector.predict_phishing_risk(test_url, 'random_forest')
        print(f"   URL: {test_url}")
        print(f"   Phishing probability: {result['phishing_probability']:.3f}")
        print(f"   Prediction: {'Phishing' if result['is_phishing'] else 'Legitimate'}")
        
        # Test batch prediction
        print("6. Testing batch prediction...")
        test_urls = [
            "https://google.com",
            "https://facebook.com", 
            "https://suspicious-site.ml",
            "https://amazon.com"
        ]
        batch_results = detector.batch_predict(test_urls, 'random_forest')
        print(f"   Analyzed {len(batch_results)} URLs")
        
        for result in batch_results:
            print(f"   - {result['url']}: {result['phishing_probability']:.3f}")
        
        print("\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
