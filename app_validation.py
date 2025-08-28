#!/usr/bin/env python3
"""
Streamlit app validation script
"""

import time
import subprocess
import requests
import sys

def check_streamlit_app():
    """Check if Streamlit app is running and accessible"""
    
    print("🚀 Checking Streamlit App Status...")
    
    try:
        # Check if the app is accessible
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Streamlit app is running and accessible!")
            print(f"📱 Local URL: http://localhost:8501")
            print(f"🌐 Access your app in browser to take screenshots")
            return True
        else:
            print(f"❌ App returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Streamlit app")
        print("💡 Make sure the app is running with: streamlit run app.py")
        return False
    except Exception as e:
        print(f"❌ Error checking app: {e}")
        return False

def test_app_components():
    """Test various components of the app"""
    
    print("\n🔧 Testing App Components...")
    
    # Import all required libraries
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import pickle
        print("✅ All required libraries imported successfully")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        return False
    
    # Check if model files exist and can be loaded
    model_files = ['diamond_price_model.pkl', 'power_transformer.pkl', 'final_dataset.pkl']
    
    for file in model_files:
        try:
            with open(file, 'rb') as f:
                pickle.load(f)
            print(f"✅ {file} loaded successfully")
        except Exception as e:
            print(f"❌ {file} failed to load: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🧪 Diamond Price Predictor - Validation Test")
    print("=" * 50)
    
    # Test components first
    if not test_app_components():
        print("\n❌ Component tests failed!")
        sys.exit(1)
    
    # Check if app is running
    app_running = check_streamlit_app()
    
    if app_running:
        print("\n✅ All tests passed! Your Streamlit app is ready.")
        print("\n📸 To capture screenshots for your README:")
        print("1. Open http://localhost:8501 in your browser")
        print("2. Take screenshots of the interface")
        print("3. Try different diamond configurations")
        print("4. Capture the prediction results")
    else:
        print("\n❌ App is not running. Start it with:")
        print("streamlit run app.py")
    
    print("\n🚀 Ready for deployment to Streamlit Cloud!")
