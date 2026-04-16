#!/usr/bin/env python3
"""Setup script for the phishing detection system."""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Phishing Detection System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run basic test
    if not run_command("python scripts/test_system.py", "Running basic system test"):
        print("❌ Basic system test failed")
        sys.exit(1)
    
    # Create necessary directories
    directories = ["logs", "assets", "data/raw", "data/processed"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the Streamlit demo: streamlit run demo/app.py")
    print("2. Train models: python scripts/train_models.py")
    print("3. Explore the Jupyter notebook: notebooks/quick_start.ipynb")
    print("\n⚠️  Remember: This is a research tool only, not for production use!")


if __name__ == "__main__":
    main()
