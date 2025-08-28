#!/usr/bin/env python3
"""
Test script to verify the diamond price prediction functionality
"""

import pandas as pd
import numpy as np
import pickle
import os

def test_diamond_prediction():
    """Test the diamond price prediction pipeline"""
    
    print("🔍 Testing Diamond Price Prediction Pipeline...")
    
    # Test data (similar to what the Streamlit app would use)
    test_input = {
        'cut': 'Premium',      # encoded as 3
        'color': 'G',          # encoded as 3  
        'clarity': 'VS1',      # encoded as 4
        'depth': 61.0,
        'table': 57.0,
        'carat': 1.0,
        'x': 6.0,
        'y': 6.0,
        'z': 3.5
    }
    
    print(f"📊 Test input: {test_input}")
    
    try:
        # Load model and transformer
        with open('diamond_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('power_transformer.pkl', 'rb') as f:
            pt = pickle.load(f)
            
        print("✅ Model and transformer loaded successfully")
        
        # Encode categorical features (same as app.py)
        cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
        color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
        clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
        
        cut_enc = cut_map[test_input['cut']]
        color_enc = color_map[test_input['color']]
        clarity_enc = clarity_map[test_input['clarity']]
        
        # Calculate PCA component (same as app.py)
        x_mean, x_std = 5.73, 1.12
        y_mean, y_std = 5.73, 1.14
        z_mean, z_std = 3.54, 0.71
        
        x_std_val = (test_input['x'] - x_mean) / x_std
        y_std_val = (test_input['y'] - y_mean) / y_std
        z_std_val = (test_input['z'] - z_mean) / z_std
        
        loading_x = 0.580088
        loading_y = 0.576372
        loading_z = 0.575581
        
        pca_1_calculated = loading_x * x_std_val + loading_y * y_std_val + loading_z * z_std_val
        
        # Create feature engineering
        carat = test_input['carat']
        carat_squared = carat ** 2
        carat_clarity = carat * clarity_enc
        cut_color = cut_enc * color_enc
        
        # Build input DataFrame
        input_features = pd.DataFrame([[
            cut_enc, color_enc, clarity_enc, test_input['depth'], test_input['table'],
            pca_1_calculated, carat, carat_squared, carat_clarity, cut_color
        ]], columns=[
            'cut', 'color', 'clarity', 'depth', 'table', 'PCA_1',
            'carat', 'carat^2', 'carat_clarity', 'cut_color'
        ])
        
        # Manual scaling (same as app.py)
        scaling_params = {
            'cut': (2.0, 1.25),
            'color': (3.0, 1.65),
            'clarity': (3.0, 2.0),
            'depth': (61.75, 1.43),
            'table': (57.46, 2.23),
            'PCA_1': (0.0, 1.0),
            'carat': (0.80, 0.47),
            'carat^2': (0.64, 0.30),
            'carat_clarity': (1.0, 0.5),
            'cut_color': (2.0, 1.0)
        }
        
        def manual_scale(df, scaling_params):
            df_scaled = df.copy()
            for col, (mean_val, std_val) in scaling_params.items():
                df_scaled[col] = (df_scaled[col] - mean_val) / std_val
            return df_scaled
        
        scaled_input = manual_scale(input_features, scaling_params)
        
        print("✅ Feature engineering completed")
        
        # Make prediction
        pred_boxcox = model.predict(scaled_input)
        pred_price = pt.inverse_transform(pred_boxcox.reshape(-1, 1))[0][0]
        
        print(f"💎 Predicted Price: ${pred_price:,.2f}")
        
        # Sanity check - price should be reasonable for a 1-carat premium diamond
        if 2000 <= pred_price <= 15000:
            print("✅ Prediction looks reasonable for a 1-carat Premium diamond")
            return True
        else:
            print(f"⚠️  Prediction seems unusual (${pred_price:,.2f})")
            return False
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

if __name__ == "__main__":
    success = test_diamond_prediction()
    if success:
        print("\n🎉 Diamond Price Prediction Pipeline is working correctly!")
    else:
        print("\n❌ There are issues with the prediction pipeline")
