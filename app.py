# diamond_price_app.py

import warnings

# Suppress version and feature name warnings from scikit-learn
warnings.filterwarnings("ignore", message="Trying to unpickle estimator PowerTransformer")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page settings
st.set_page_config(
    page_title='Diamond Price Prediction',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.title('Diamond Price Predictor')

# -------------------
# Sidebar: User Inputs
# -------------------
st.sidebar.header('Enter Diamond Features')

# Categorical features
cut = st.sidebar.selectbox('Cut', options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.sidebar.selectbox('Color', options=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.sidebar.selectbox('Clarity', options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

# Other numeric features
depth = st.sidebar.slider('Depth (%)', 43.0, 79.0, 61.0)
table = st.sidebar.slider('Table (%)', 43.0, 95.0, 57.0)
carat = st.sidebar.slider('Carat', 0.2, 5.1, 0.9)

# Raw size measurements (for PCA computation)
x_factor_input = st.sidebar.slider('x (Premium)', 0.0, 10.0, 5.73)
y_factor_input = st.sidebar.slider('y (Good)', 0.0, 10.0, 5.73)
z_factor_input = st.sidebar.slider('z (Very Good)', 0.0, 10.0, 3.54)

# -----------------------------
# Compute the PCA component
# -----------------------------
# Fixed mean and std values obtained during training (placeholders; update with your actual values)
x_mean, x_std = 5.73, 1.12
y_mean, y_std = 5.73, 1.14
z_mean, z_std = 3.54, 0.71

# Standardize raw inputs
x_std_val = (x_factor_input - x_mean) / x_std
y_std_val = (y_factor_input - y_mean) / y_std
z_std_val = (z_factor_input - z_mean) / z_std

# PCA loadings from training (placeholders; update with your actual values)
loading_x = 0.580088
loading_y = 0.576372
loading_z = 0.575581

# Calculate the PCA component
pca_1_calculated = loading_x * x_std_val + loading_y * y_std_val + loading_z * z_std_val

# -----------------------------
# Encode categorical features
# -----------------------------
cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

cut_enc = cut_map[cut]
color_enc = color_map[color]
clarity_enc = clarity_map[clarity]

# -----------------------------
# Compute derived features
# -----------------------------
carat_squared = carat ** 2
carat_clarity = carat * clarity_enc
cut_color = cut_enc * color_enc

# -----------------------------
# Build the final input DataFrame
# -----------------------------
input_features = pd.DataFrame([[
    cut_enc, color_enc, clarity_enc, depth, table,
    pca_1_calculated,   # computed PCA component
    carat, carat_squared, carat_clarity, cut_color
]], columns=[
    'cut', 'color', 'clarity', 'depth', 'table', 'PCA_1',
    'carat', 'carat^2', 'carat_clarity', 'cut_color'
])

# --------------------------------------------------------
# Manual Scaling: Use predetermined means and standard deviations
# (Replace these values with the ones calculated from your training set)
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

# --------------------------------------------------------
# Load Model and PowerTransformer using absolute paths
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "diamond_price_model.pkl")
pt_path = os.path.join(current_dir, "power_transformer.pkl")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
with open(pt_path, 'rb') as pt_file:
    pt = pickle.load(pt_file)

# Make prediction (model was trained on Box-Cox transformed target)
pred_boxcox = model.predict(scaled_input)
pred_price = pt.inverse_transform(pred_boxcox.reshape(-1, 1))[0][0]

# -----------------------------
# Display Prediction
# -----------------------------
st.subheader('Prediction')
st.write('Based on your inputs, the predicted diamond price is:')
st.metric(label="Estimated Price (USD)", value=f"${pred_price:,.2f}")

# -----------------------------
# Display Historical Price Chart (Optional)
# -----------------------------
st.subheader("Historical Diamond Price Distribution")
try:
    dataset_path = os.path.join(current_dir, "final_dataset.pkl")
    final_dataset = pd.read_pickle(dataset_path)
    # Inverse-transform the Box-Cox target to get original price values
    original_prices = pt.inverse_transform(final_dataset['price_boxcox'].values.reshape(-1, 1))
    price_series = pd.Series(original_prices.flatten(), name="Price").sort_values().reset_index(drop=True)
    st.line_chart(price_series)
except Exception as e:
    st.write("Historical price distribution data not available.", e)
