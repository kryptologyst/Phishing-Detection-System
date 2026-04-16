"""Training script for the phishing detection system."""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.phishing_detector import PhishingDetector


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train phishing detection models")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    print(f"Training Phishing Detection Models")
    print(f"Samples: {args.samples}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    try:
        # Initialize detector
        detector = PhishingDetector(args.config)
        
        # Run full pipeline
        print("Running full training pipeline...")
        results = detector.run_full_pipeline(args.samples)
        
        # Add timestamp
        results['training_timestamp'] = datetime.now().isoformat()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Results saved to: {args.output}")
        
        # Print summary
        print("\nModel Performance Summary:")
        print("-" * 30)
        
        if 'leaderboard' in results:
            for metric, rankings in results['leaderboard'].items():
                if rankings:
                    best_model = rankings[0]
                    print(f"{metric}: {best_model['model']} ({best_model['score']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
