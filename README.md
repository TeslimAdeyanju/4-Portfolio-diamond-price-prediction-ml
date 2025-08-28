# 💎 Diamond Price Prediction: The 4C Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end machine learning project that predicts diamond prices using the industry-standard 4C model (Cut, Color, Clarity, Carat) with advanced feature engineering and model optimization.**

## 🎯 Project Overview

This project demonstrates a comprehensive machine learning pipeline for predicting diamond prices based on their physical and qualitative attributes. Using a dataset of 53,940 diamonds, the model achieves high accuracy through sophisticated feature engineering, including Principal Component Analysis (PCA), polynomial features, and advanced preprocessing techniques.

### ✨ Key Features

- **Complete ML Pipeline**: From data preprocessing to model deployment
- **Advanced Feature Engineering**: PCA, polynomial features, and categorical encoding
- **Multiple Model Comparison**: Decision Trees, Random Forest, XGBoost, and more
- **Interactive Web App**: Streamlit-based prediction interface
- **Production Ready**: Pickle-serialized models with error handling

## 🚀 Live Demo

🔗 **[Try the Diamond Price Predictor](your-streamlit-app-url-here)**

## 📊 Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **XGBoost (Best)** | 0.982 | $486 | $298 |
| Random Forest | 0.978 | $542 | $321 |
| Gradient Boosting | 0.975 | $578 | $334 |

## 🛠️ Tech Stack

- **Python 3.8+**
- **Data Analysis**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Deployment**: Streamlit
- **Statistical Analysis**: scipy, statsmodels

## 📂 Project Structure

```
diamond-price-prediction/
│
├── 📊 Diamond_price_prediction_note.ipynb  # Complete analysis notebook
├── 🚀 app.py                              # Streamlit web application
├── 🔧 requirements.txt                     # Project dependencies
├── 📈 data.csv                            # Raw dataset
├── 🤖 xgboost_diamond_model.pkl           # Trained XGBoost model
├── 🔄 power_transformer.pkl               # Feature transformer
├── 💾 final_dataset.pkl                   # Processed dataset
└── 📋 README.md                           # Project documentation
```

## 🎯 Key Insights

### Diamond Pricing Factors
1. **Carat Weight** (r=0.92): Strongest predictor of price
2. **Size Dimensions** (PCA): Combined x, y, z measurements
3. **Cut Quality**: Premium and Ideal cuts command higher prices
4. **Color Grade**: D (colorless) to J (near colorless) scale
5. **Clarity**: FL (flawless) to I3 (included) scale

### Feature Engineering Highlights
- **PCA Components**: Reduced 3D measurements to principal components
- **Polynomial Features**: Captured non-linear relationships
- **Categorical Encoding**: Ordinal encoding for quality grades
- **Target Transformation**: Power transformation for price normalization

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/TeslimAdeyanju/diamond-price-prediction-ml.git
cd diamond-price-prediction-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

### 4. Explore the Jupyter Notebook
```bash
jupyter notebook Diamond_price_prediction_note.ipynb
```

## 📱 Using the Web App

1. **Select Diamond Attributes**: Use the sidebar to input diamond specifications
2. **Adjust Parameters**: Fine-tune cut, color, clarity, carat, and dimensions
3. **Get Prediction**: Instantly see the predicted price with confidence intervals
4. **Explore Insights**: View feature importance and model explanations

## 🔬 Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Handled missing values and outliers
- **Feature Scaling**: StandardScaler for numerical features
- **Encoding**: Ordinal encoding for categorical variables

### 2. Exploratory Data Analysis
- **Statistical Analysis**: Chi-square tests for independence
- **Visualization**: Correlation matrices, distribution plots
- **Feature Relationships**: Multicollinearity detection

### 3. Feature Engineering
- **PCA**: Dimensionality reduction for size measurements
- **Polynomial Features**: Interaction terms for better accuracy
- **Feature Selection**: K-best selection and correlation analysis

### 4. Model Development
- **Baseline Models**: Linear regression benchmarking
- **Advanced Models**: Tree-based and ensemble methods
- **Hyperparameter Tuning**: Grid search optimization
- **Cross-Validation**: 5-fold CV for robust evaluation

## 📈 Results & Analysis

The XGBoost model achieved the best performance with an R² of 0.982, demonstrating exceptional accuracy in diamond price prediction. Key findings include:

- **Carat weight** is the most influential factor (92% correlation with price)
- **Premium and Ideal cuts** significantly impact pricing
- **Color and clarity** have moderate but important effects
- **Size dimensions** when combined via PCA improve predictions

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- **End-to-End ML Pipeline**: From data exploration to deployment
- **Advanced Feature Engineering**: PCA, polynomial features, encoding strategies
- **Model Selection & Tuning**: Comparative analysis of multiple algorithms
- **Statistical Analysis**: Hypothesis testing and correlation analysis
- **Web Development**: Interactive Streamlit applications
- **Production Deployment**: Model serialization and error handling

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Neural networks for complex patterns
- [ ] **Real-time Data**: Integration with diamond market APIs
- [ ] **Advanced UI**: Enhanced Streamlit interface with visualizations
- [ ] **Model Explainability**: SHAP values for prediction explanations
- [ ] **A/B Testing**: Multiple model comparison in production

## 👨‍💻 About the Author

**Teslim Uthman Adeyanju**  
*Data Scientist & Machine Learning Engineer*

- 📧 **Email**: [info@adeyanjuteslim.co.uk](mailto:info@adeyanjuteslim.co.uk)
- 🔗 **LinkedIn**: [linkedin.com/in/adeyanjuteslimuthman](https://www.linkedin.com/in/adeyanjuteslimuthman)
- 🌐 **Website**: [adeyanjuteslim.co.uk](https://adeyanjuteslim.co.uk)
- 📊 **Portfolio**: More projects on [GitHub](https://github.com/TeslimAdeyanju)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Kaggle - Type of the Diamond](https://www.kaggle.com/datasets/willianoliveiragibin/type-of-the-diamond)
- **Domain Knowledge**: Gemological Institute of America (GIA) diamond grading standards
- **Tools**: The amazing Python data science ecosystem

---

⭐ **If you found this project helpful, please give it a star!**
