# 🌍 World Happiness Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)

**An Intelligent Systems Project for Predicting Country Happiness Scores**

</div>

---

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Student Name** | Abdelrahman Mahmoud Seada |
| **Student ID** | 201801343 |
| **Course** | Artificial Intelligence |
| **Instructor** | Lamiaa Khairy |

---

## 📖 Overview

This project implements a **Machine Learning model** to predict country happiness scores using the World Happiness Report dataset. The system uses **Linear Regression** with comprehensive data preprocessing, feature engineering, and model evaluation techniques.

### 🎯 Objectives

1. **Analyze** the World Happiness Report dataset to understand factors affecting happiness
2. **Build** a predictive model using Linear Regression
3. **Evaluate** model performance using multiple metrics
4. **Visualize** data insights and model results
5. **Enable** interactive predictions for new data

---

## 📊 Dataset

### Source
- **Dataset**: [World Happiness Ranking Dataset](https://www.kaggle.com/datasets/nalisha/world-happiness-ranking-dataset)
- **Records**: 1,232 entries
- **Time Period**: 2015-2020

### Features Used

| Feature | Description |
|---------|-------------|
| Economy (GDP per Capita) | Economic output per person |
| Family | Social support and family bonds |
| Health (Life Expectancy) | Average life expectancy |
| Freedom | Personal freedom to make choices |
| Trust (Government Corruption) | Perception of government integrity |
| Generosity | Charitable giving behavior |

### Target Variable
- **Happiness Score**: Numerical score (0-10) representing overall life satisfaction

---

## 🏗️ Project Structure

```
Project/
├── app.py                                       # Premium Web Dashboard (Streamlit)
├── intelligent_systems_project_201801343.py    # Main Python script
├── Intelligent_Systems_Project_201801343.ipynb # Jupyter notebook version
├── README.md                                    # Project documentation
├── requirements.txt                             # Python dependencies
├── dataset/
│   └── world_happiness_report.csv              # Dataset file
└── Explaining Video/                            # Project demo video
```

---

## 🔧 Installation & Requirements

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Required Libraries

```bash
pip install -r requirements.txt
```

### Dependencies List

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3.0 | Data manipulation |
| numpy | ≥1.21.0 | Numerical computing |
| matplotlib | ≥3.4.0 | Data visualization |
| seaborn | ≥0.11.0 | Statistical plots |
| scikit-learn | ≥1.0.0 | Machine learning |
| **streamlit** | ≥1.20.0 | **Web Dashboard** |
| **plotly** | ≥5.0.0 | **Interactive Charts** |

---

## 🚀 How to Run

### Option 1: Premium Web Dashboard (Recommended) ⭐

```bash
# Navigate to project directory
cd "d:\College\2026 - Fall\AI\Project"

# Launch the interactive dashboard
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

**Features:**
- 🎨 Premium dark grey theme with modern UI
- 📊 Interactive Plotly visualizations
- 🔮 Real-time happiness prediction with sliders
- 📈 Detailed analysis with tabs
- 📋 Data explorer with download option

### Option 2: Run the Python Script

```bash
python intelligent_systems_project_201801343.py
```

### Option 3: Run in Jupyter Notebook

```bash
jupyter notebook
# Open: Intelligent_Systems_Project_201801343.ipynb
```

---

## 📈 Methodology

### 1. Data Preprocessing

```
Raw Data → Missing Value Analysis → KNN Imputation → Feature Selection
```

- **Target Handling**: Rows with missing happiness scores are dropped
- **Feature Imputation**: KNN Imputer (k=5) fills missing feature values
- **No Scaling**: Linear Regression doesn't require feature scaling

### 2. Model Training

- **Algorithm**: Linear Regression
- **Train/Test Split**: 80/20 ratio
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Random State**: 42 (for reproducibility)

### 3. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| R² Score | Proportion of variance explained (higher is better) |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| Cross-Validation | K-fold validation for generalization |

### 4. Classification Analysis

Happiness scores are categorized for detailed analysis:

| Category | Score Range |
|----------|-------------|
| 🔴 Low | < 4.5 |
| 🟡 Medium | 4.5 - 6.0 |
| 🟢 High | > 6.0 |

---

## 📊 Visualizations

The project generates the following visualizations:

1. **Correlation Heatmap**: Shows relationships between features
2. **Feature Importance**: Bar chart of regression coefficients
3. **Actual vs Predicted**: Scatter plot comparing true and predicted values
4. **Residual Analysis**: Plots to assess model assumptions
5. **Confusion Matrix**: Classification performance visualization

---

## 🎮 Interactive Prediction

The script includes an interactive mode where you can input feature values and get happiness predictions:

```
  Economy/GDP: 1.2
  Family Support: 1.1
  Health/Life Expectancy: 0.9
  Freedom: 0.5
  Trust/Low Corruption: 0.3
  Generosity: 0.2

  🎯 Predicted Happiness Score: 6.15
  📊 Category: High
```

---

## 📝 Key Findings

### Feature Impact Analysis

1. **Economy (GDP)**: Strongest positive correlation with happiness
2. **Family Support**: Second most influential factor
3. **Health**: Significant positive impact
4. **Freedom**: Moderate positive effect
5. **Trust**: Important but smaller effect
6. **Generosity**: Smallest but still positive impact

### Model Performance

- The Linear Regression model achieves strong predictive performance
- Cross-validation ensures the model generalizes well to unseen data
- Classification accuracy provides insights into categorical predictions

---

## 🔮 Future Improvements

- [ ] Implement additional ML algorithms (Random Forest, XGBoost)
- [ ] Add feature engineering (interaction terms, polynomial features)
- [ ] Create web-based interface using Flask/Streamlit
- [ ] Incorporate time-series analysis for trend prediction
- [ ] Add geographical visualization with world maps

---

## 📚 References

1. World Happiness Report - United Nations Sustainable Development Solutions Network
2. Scikit-learn Documentation - https://scikit-learn.org/
3. Pandas Documentation - https://pandas.pydata.org/
4. Helliwell, J. F., Layard, R., & Sachs, J. (2020). World Happiness Report 2020

---

## 📄 License

This project is for educational purposes as part of the AI course curriculum.

---

## 🤝 Acknowledgments

- **Instructor**: Lamiaa Khairy - For guidance and course materials
- **Dataset Source**: Kaggle - World Happiness Ranking Dataset
- **University**: For providing the educational platform

---

<div align="center">

**Made with ❤️ by Abdelrahman Mahmoud Seada**

*Student ID: 201801343*

</div>

