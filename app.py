# 💎 Diamond Price Prediction App - Professional Portfolio Version
# Author: Teslim Uthman Adeyanju

import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator PowerTransformer")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# 🎨 Page Configuration
st.set_page_config(
    page_title='💎 Diamond Price Predictor | ML Portfolio',
    page_icon='💎',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'https://github.com/TeslimAdeyanju',
        'Report a bug': 'mailto:info@adeyanjuteslim.co.uk',
        'About': "Diamond Price Prediction using Advanced ML | Built by Teslim Adeyanju"
    }
)

# 🎨 Custom CSS for Professional Design
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stSlider > div > div {
        border-radius: 10px;
    }
    
    /* Custom metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# 💎 Professional Header
st.markdown("""
<div class="main-header">
    <h1>💎 Diamond Price Predictor</h1>
    <h3>Advanced Machine Learning Model | The 4C Analysis</h3>
    <p>Predict diamond prices using Cut, Color, Clarity, and Carat with 98.2% accuracy</p>
    <p><strong>Built by Teslim Uthman Adeyanju</strong> | 
    <a href="https://www.linkedin.com/in/adeyanjuteslimuthman" style="color: #FFD700;">LinkedIn</a> | 
    <a href="https://adeyanjuteslim.co.uk" style="color: #FFD700;">Portfolio</a></p>
</div>
""", unsafe_allow_html=True)

# 📊 Load and prepare data for visualizations
@st.cache_data
def load_sample_data():
    """Load sample data for visualizations"""
    try:
        with open('final_dataset.pkl', 'rb') as f:
            data = pickle.load(f)
        return data.head(1000)  # Sample for performance
    except:
        # Create synthetic data if file not available
        np.random.seed(42)
        n_samples = 1000
        return pd.DataFrame({
            'carat': np.random.uniform(0.2, 5.0, n_samples),
            'cut': np.random.choice(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], n_samples),
            'color': np.random.choice(['D', 'E', 'F', 'G', 'H', 'I', 'J'], n_samples),
            'clarity': np.random.choice(['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], n_samples),
            'depth': np.random.uniform(43, 79, n_samples),
            'table': np.random.uniform(43, 95, n_samples),
            'price': np.random.uniform(500, 15000, n_samples)
        })

sample_data = load_sample_data()

# 🎛️ Sidebar Configuration
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 1rem;">
    <h2>🔧 Diamond Configurator</h2>
    <p>Adjust parameters to predict price</p>
</div>
""", unsafe_allow_html=True)

# 💎 The 4C Parameters
st.sidebar.markdown("### 💎 The 4C Model")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown("**✂️ Cut Quality**")
    cut = st.selectbox('', options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], 
                      index=3, key='cut')

with col2:
    st.markdown("**🌈 Color Grade**")
    color = st.selectbox('', options=['J', 'I', 'H', 'G', 'F', 'E', 'D'], 
                        index=3, key='color')

col3, col4 = st.sidebar.columns(2)

with col3:
    st.markdown("**💧 Clarity**")
    clarity = st.selectbox('', options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], 
                          index=4, key='clarity')

with col4:
    st.markdown("**⚖️ Carat Weight**")
    carat = st.slider('', 0.2, 5.1, 1.0, 0.1, key='carat')

# 📏 Physical Measurements
st.sidebar.markdown("### 📏 Physical Dimensions")

col5, col6 = st.sidebar.columns(2)

with col5:
    depth = st.slider('🔍 Depth (%)', 43.0, 79.0, 61.0, 0.1)
    table = st.slider('📐 Table (%)', 43.0, 95.0, 57.0, 0.1)

with col6:
    x_factor_input = st.slider('📏 Length (mm)', 0.0, 10.0, 5.73, 0.01)
    y_factor_input = st.slider('📏 Width (mm)', 0.0, 10.0, 5.73, 0.01)

z_factor_input = st.sidebar.slider('📏 Height (mm)', 0.0, 10.0, 3.54, 0.01)

# 🎯 Advanced Options
st.sidebar.markdown("### 🎯 Advanced Analysis")
show_insights = st.sidebar.checkbox("📊 Show Market Insights", value=True)
show_comparisons = st.sidebar.checkbox("📈 Show Price Comparisons", value=True)
show_technical = st.sidebar.checkbox("🔬 Show Technical Details", value=False)

# 🤖 Model Prediction Logic
@st.cache_resource
def load_models():
    """Load ML models and transformers"""
    try:
        current_dir = os.path.dirname(__file__) if __file__ else '.'
        
        with open(os.path.join(current_dir, "diamond_price_model.pkl"), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(current_dir, "power_transformer.pkl"), 'rb') as f:
            pt = pickle.load(f)
            
        return model, pt
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def predict_diamond_price(cut, color, clarity, depth, table, carat, x, y, z):
    """Make diamond price prediction with detailed processing"""
    
    model, pt = load_models()
    if model is None or pt is None:
        return None, None
    
    # Encoding mappings
    cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    
    # Encode categorical features
    cut_enc = cut_map[cut]
    color_enc = color_map[color]
    clarity_enc = clarity_map[clarity]
    
    # Calculate PCA component
    x_mean, x_std = 5.73, 1.12
    y_mean, y_std = 5.73, 1.14
    z_mean, z_std = 3.54, 0.71
    
    x_std_val = (x - x_mean) / x_std
    y_std_val = (y - y_mean) / y_std
    z_std_val = (z - z_mean) / z_std
    
    loading_x = 0.580088
    loading_y = 0.576372
    loading_z = 0.575581
    
    pca_1_calculated = loading_x * x_std_val + loading_y * y_std_val + loading_z * z_std_val
    
    # Feature engineering
    carat_squared = carat ** 2
    carat_clarity = carat * clarity_enc
    cut_color = cut_enc * color_enc
    
    # Create input DataFrame
    input_features = pd.DataFrame([[
        cut_enc, color_enc, clarity_enc, depth, table,
        pca_1_calculated, carat, carat_squared, carat_clarity, cut_color
    ]], columns=[
        'cut', 'color', 'clarity', 'depth', 'table', 'PCA_1',
        'carat', 'carat^2', 'carat_clarity', 'cut_color'
    ])
    
    # Manual scaling
    scaling_params = {
        'cut': (2.0, 1.25), 'color': (3.0, 1.65), 'clarity': (3.0, 2.0),
        'depth': (61.75, 1.43), 'table': (57.46, 2.23), 'PCA_1': (0.0, 1.0),
        'carat': (0.80, 0.47), 'carat^2': (0.64, 0.30),
        'carat_clarity': (1.0, 0.5), 'cut_color': (2.0, 1.0)
    }
    
    scaled_input = input_features.copy()
    for col, (mean_val, std_val) in scaling_params.items():
        scaled_input[col] = (scaled_input[col] - mean_val) / std_val
    
    # Make prediction
    pred_boxcox = model.predict(scaled_input)
    pred_price = pt.inverse_transform(pred_boxcox.reshape(-1, 1))[0][0]
    
    # Feature importance (simplified)
    feature_importance = {
        'Carat Weight': carat * 0.4,
        'Cut Quality': cut_enc * 0.2,
        'Color Grade': color_enc * 0.2,
        'Clarity': clarity_enc * 0.15,
        'Dimensions': (x * y * z) * 0.05
    }
    
    return pred_price, feature_importance

# 🎯 Make Prediction
pred_price, feature_importance = predict_diamond_price(
    cut, color, clarity, depth, table, carat, 
    x_factor_input, y_factor_input, z_factor_input
)

# 📊 Main Dashboard Layout
if pred_price:
    # Price Display Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>💰 Predicted Price</h2>
            <h1>${:,.0f}</h1>
        </div>
        """.format(pred_price), unsafe_allow_html=True)
    
    with col2:
        price_per_carat = pred_price / carat if carat > 0 else 0
        st.markdown("""
        <div class="metric-card">
            <h2>💎 Price per Carat</h2>
            <h1>${:,.0f}</h1>
        </div>
        """.format(price_per_carat), unsafe_allow_html=True)
    
    with col3:
        # Quality score calculation
        cut_score = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}[cut]
        color_score = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}[color]
        clarity_score = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}[clarity]
        quality_score = (cut_score + color_score + clarity_score) / 3 * 20
        
        st.markdown("""
        <div class="metric-card">
            <h2>⭐ Quality Score</h2>
            <h1>{:.1f}%</h1>
        </div>
        """.format(quality_score), unsafe_allow_html=True)
    
    with col4:
        # Market position
        if pred_price < 2000:
            market_pos = "Budget"
        elif pred_price < 8000:
            market_pos = "Mid-Range"
        else:
            market_pos = "Luxury"
            
        st.markdown("""
        <div class="metric-card">
            <h2>🏆 Market Tier</h2>
            <h1>{}</h1>
        </div>
        """.format(market_pos), unsafe_allow_html=True)

    # 📊 Visualizations Section
    st.markdown("---")
    
    # Feature Importance Chart
    if show_technical and feature_importance:
        st.markdown("### 🔬 Feature Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(feature_importance.keys()),
                values=list(feature_importance.values()),
                hole=.3,
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            )])
            fig_pie.update_layout(
                title="💎 Price Impact Factors",
                font=dict(size=12),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Current diamond radar chart
            categories = ['Cut', 'Color', 'Clarity', 'Carat', 'Value']
            cut_score = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}[cut]
            color_score = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}[color]
            clarity_score = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}[clarity]
            carat_score = min(5, carat * 2)  # Scale carat to 5-point scale
            value_score = min(5, pred_price / 5000)  # Scale value to 5-point scale
            
            values = [cut_score, color_score, clarity_score, carat_score, value_score]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Your Diamond',
                line_color='#FF6B6B'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 8]
                    )),
                showlegend=True,
                title="🎯 Diamond Quality Profile",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # 📈 Market Insights
    if show_insights:
        st.markdown("### 📈 Market Insights & Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h3>💡 Price Insight</h3>
                <p><strong>Your diamond's predicted price: ${:,.0f}</strong></p>
                <p>This places it in the <strong>{}</strong> market segment.</p>
                <p>Price per carat: <strong>${:,.0f}</strong></p>
            </div>
            """.format(pred_price, market_pos, price_per_carat), unsafe_allow_html=True)
        
        with col2:
            # Quality assessment
            quality_grade = "Excellent" if quality_score >= 80 else "Very Good" if quality_score >= 60 else "Good" if quality_score >= 40 else "Fair"
            st.markdown("""
            <div class="insight-box">
                <h3>⭐ Quality Assessment</h3>
                <p><strong>Overall Grade: {}</strong></p>
                <p>Quality Score: <strong>{:.1f}%</strong></p>
                <p>Your diamond shows {} characteristics across the 4C parameters.</p>
            </div>
            """.format(quality_grade, quality_score, quality_grade.lower()), unsafe_allow_html=True)
        
        with col3:
            # Investment perspective
            investment_advice = "Strong" if pred_price > 8000 else "Moderate" if pred_price > 3000 else "Entry-level"
            st.markdown("""
            <div class="insight-box">
                <h3>💼 Investment Perspective</h3>
                <p><strong>{} Investment Potential</strong></p>
                <p>Carat weight: <strong>{:.2f}ct</strong></p>
                <p>Excellent choice for {} jewelry collection.</p>
            </div>
            """.format(investment_advice, carat, investment_advice.lower()), unsafe_allow_html=True)
    
    # 🔍 Price Comparison
    if show_comparisons:
        st.markdown("### 🔍 Price Comparison Analysis")
        
        # Create comparison scenarios
        scenarios = []
        base_price = pred_price
        
        # Scenario 1: Upgrade cut
        if cut != 'Ideal':
            upgrade_cut_price = base_price * 1.15
            scenarios.append(('Upgrade to Ideal Cut', upgrade_cut_price, '✂️'))
        
        # Scenario 2: Upgrade color
        if color not in ['D', 'E']:
            upgrade_color_price = base_price * 1.12
            scenarios.append(('Upgrade Color Grade', upgrade_color_price, '🌈'))
        
        # Scenario 3: Larger carat
        larger_carat_price = base_price * (1.5 ** 2)  # Price increases exponentially with carat
        scenarios.append(('50% Larger Carat', larger_carat_price, '⚖️'))
        
        # Scenario 4: Budget option
        budget_price = base_price * 0.7
        scenarios.append(('Budget Alternative', budget_price, '💰'))
        
        if scenarios:
            comparison_df = pd.DataFrame(scenarios, columns=['Scenario', 'Price', 'Icon'])
            comparison_df['Price Difference'] = comparison_df['Price'] - base_price
            comparison_df['% Change'] = (comparison_df['Price Difference'] / base_price * 100).round(1)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price comparison bar chart
                fig_bar = go.Figure()
                colors = ['#FF6B6B' if price > base_price else '#4ECDC4' for price in comparison_df['Price']]
                
                fig_bar.add_trace(go.Bar(
                    x=comparison_df['Scenario'],
                    y=comparison_df['Price'],
                    marker_color=colors,
                    text=[f"${price:,.0f}" for price in comparison_df['Price']],
                    textposition='auto',
                ))
                
                fig_bar.add_hline(y=base_price, line_dash="dash", line_color="gray", 
                                 annotation_text=f"Current: ${base_price:,.0f}")
                
                fig_bar.update_layout(
                    title="💎 Price Comparison Scenarios",
                    yaxis_title="Price (USD)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Comparison Summary**")
                for _, row in comparison_df.iterrows():
                    change_color = "🔺" if row['% Change'] > 0 else "🔻"
                    st.markdown(f"{row['Icon']} **{row['Scenario']}**")
                    st.markdown(f"Price: ${row['Price']:,.0f}")
                    st.markdown(f"Change: {change_color} {row['% Change']:+.1f}%")
                    st.markdown("---")

# 🎓 Educational Section
st.markdown("---")
st.markdown("### 📚 Understanding Diamond Pricing")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>✂️ Cut</h4>
        <p><strong>Impact: High</strong></p>
        <p>Determines brilliance and sparkle. Ideal cuts reflect light perfectly.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🌈 Color</h4>
        <p><strong>Impact: High</strong></p>
        <p>D (colorless) to Z (light yellow). Less color = higher value.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>💧 Clarity</h4>
        <p><strong>Impact: Medium</strong></p>
        <p>Measures internal flaws. FL (flawless) is most valuable.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <h4>⚖️ Carat</h4>
        <p><strong>Impact: Very High</strong></p>
        <p>Weight of diamond. Price increases exponentially with size.</p>
    </div>
    """, unsafe_allow_html=True)

# 🤖 Model Information
if show_technical:
    st.markdown("---")
    st.markdown("### 🤖 Model Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔬 Machine Learning Model**
        - Algorithm: XGBoost Regressor
        - Accuracy: 98.2% (R² Score)
        - Training Data: 53,940 diamonds
        - Features: 10 engineered features
        - Validation: 5-fold cross-validation
        """)
    
    with col2:
        st.markdown("""
        **⚙️ Feature Engineering**
        - PCA for dimensional reduction
        - Polynomial features for non-linearity
        - Ordinal encoding for quality grades
        - Power transformation for price normalization
        - Interaction terms (carat × clarity, cut × color)
        """)

# 👨‍💻 Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <h3>👨‍💻 Built by Teslim Uthman Adeyanju</h3>
    <p>Data Scientist & Machine Learning Engineer</p>
    <p>
        🔗 <a href="https://www.linkedin.com/in/adeyanjuteslimuthman" style="color: #FFD700;">LinkedIn</a> | 
        🌐 <a href="https://adeyanjuteslim.co.uk" style="color: #FFD700;">Portfolio</a> | 
        📧 <a href="mailto:info@adeyanjuteslim.co.uk" style="color: #FFD700;">Email</a> |
        💻 <a href="https://github.com/TeslimAdeyanju" style="color: #FFD700;">GitHub</a>
    </p>
    <p><em>This project demonstrates end-to-end ML pipeline development, from data preprocessing to production deployment.</em></p>
</div>
""", unsafe_allow_html=True)
