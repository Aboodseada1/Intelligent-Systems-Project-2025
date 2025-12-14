"""
=====================================================================
    WORLD HAPPINESS PREDICTION - PREMIUM DASHBOARD
=====================================================================
    
    Student: Abdelrahman Mahmoud Seada (201801343)
    Instructor: Lamiaa Khairy
    Course: Artificial Intelligence
    
    A modern, premium dark-themed dashboard for happiness prediction
=====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    confusion_matrix, 
    classification_report
)

# ============================================================================
#                           PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="World Happiness Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
#                           CUSTOM DARK THEME CSS
# ============================================================================

st.markdown("""
<style>
    /* Main background - Premium Dark Grey */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #151525 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #2d2d44 0%, #1f1f35 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
        font-size: 0.9rem !important;
    }
    
    /* Cards/Containers */
    .premium-card {
        background: linear-gradient(135deg, #252538 0%, #1a1a2e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
    }
    
    /* Glowing accent */
    .glow-text {
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
    }
    
    /* Success message */
    .success-badge {
        background: linear-gradient(135deg, #00c853 0%, #00a844 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0,200,83,0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.6);
    }
    
    /* Input fields */
    .stNumberInput input, .stSelectbox select {
        background: #2a2a3e !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Slider */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30,30,47,0.8);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0a0a0;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        background: #1e1e2f !important;
        border-radius: 10px;
    }
    
    /* Hero section */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .hero-subtitle {
        color: #a0a0a0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #606080;
        padding: 30px;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin-top: 50px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #2a2a3e;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
#                           CONFIGURATION
# ============================================================================

class Config:
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATASET_PATH = SCRIPT_DIR / "dataset" / "world_happiness_report.csv"
    
    TARGET = 'Happiness Score'
    FEATURES = [
        'Economy (GDP per Capita)',
        'Family',
        'Health (Life Expectancy)',
        'Freedom',
        'Trust (Government Corruption)',
        'Generosity'
    ]
    
    LOW_THRESHOLD = 4.5
    MEDIUM_THRESHOLD = 6.0

# ============================================================================
#                           PLOTLY DARK THEME
# ============================================================================

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#ffffff', 'family': 'Inter, sans-serif'},
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.1)'
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.1)'
        }
    }
}

COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#00d4ff',
    'success': '#00c853',
    'warning': '#ffab00',
    'danger': '#ff5252',
    'gradient': ['#667eea', '#764ba2', '#00d4ff', '#00c853', '#ffab00', '#ff5252']
}

# ============================================================================
#                           DATA FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    df = pd.read_csv(Config.DATASET_PATH)
    return df

@st.cache_data
def prepare_data(df):
    """Clean and prepare data for modeling."""
    df_clean = df.dropna(subset=[Config.TARGET]).copy()
    
    available_features = [f for f in Config.FEATURES if f in df_clean.columns]
    X = df_clean[available_features].copy()
    y = df_clean[Config.TARGET].copy()
    
    if X.isnull().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        X_filled = imputer.fit_transform(X)
        X = pd.DataFrame(X_filled, columns=available_features, index=X.index)
    
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y, df_clean

@st.cache_resource
def train_model(X, y):
    """Train and cache the model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    return model, X_train, X_test, y_train, y_test, y_pred, metrics

def classify_happiness(score):
    if score < Config.LOW_THRESHOLD:
        return 'Low'
    elif score < Config.MEDIUM_THRESHOLD:
        return 'Medium'
    else:
        return 'High'

# ============================================================================
#                           VISUALIZATION FUNCTIONS
# ============================================================================

def create_correlation_heatmap(X, y):
    """Create premium correlation heatmap."""
    data = X.copy()
    data[Config.TARGET] = y
    corr_matrix = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
            [0, '#ff5252'],
            [0.5, '#2d2d44'],
            [1, '#00d4ff']
        ],
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={'size': 11, 'color': 'white'},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': '📊 Feature Correlation Matrix', 'font': {'size': 20}},
        height=500,
        **PLOTLY_TEMPLATE['layout']
    )
    
    return fig

def create_feature_importance(model, feature_names):
    """Create premium feature importance chart."""
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=True)
    
    colors = [COLORS['danger'] if x < 0 else COLORS['success'] for x in importance['Coefficient']]
    
    fig = go.Figure(go.Bar(
        x=importance['Coefficient'],
        y=importance['Feature'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        hovertemplate='%{y}<br>Coefficient: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': '📈 Feature Importance', 'font': {'size': 20}},
        xaxis_title='Coefficient Value',
        height=400,
        **PLOTLY_TEMPLATE['layout']
    )
    
    return fig

def create_actual_vs_predicted(y_test, y_pred, r2):
    """Create premium actual vs predicted scatter plot."""
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=12,
            color=COLORS['accent'],
            opacity=0.7,
            line=dict(color='white', width=1)
        ),
        name='Predictions',
        hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color=COLORS['danger'], dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f'R² = {r2:.4f}',
        showarrow=False,
        font=dict(size=16, color=COLORS['accent']),
        bgcolor='rgba(0,0,0,0.7)',
        borderpad=10,
        bordercolor=COLORS['accent'],
        borderwidth=1
    )
    
    fig.update_layout(
        title={'text': '🎯 Actual vs Predicted', 'font': {'size': 20}},
        xaxis_title='Actual Happiness Score',
        yaxis_title='Predicted Happiness Score',
        height=500,
        **PLOTLY_TEMPLATE['layout']
    )
    
    return fig

def create_residual_plots(y_test, y_pred):
    """Create premium residual analysis plots."""
    residuals = y_test - y_pred
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Residuals vs Predicted', 'Residual Distribution'])
    
    # Residuals vs Predicted
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(size=10, color=COLORS['secondary'], opacity=0.7),
        name='Residuals'
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['danger'], row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=25,
        marker=dict(color=COLORS['accent'], line=dict(color='white', width=1)),
        name='Distribution'
    ), row=1, col=2)
    
    fig.add_vline(x=0, line_dash='dash', line_color=COLORS['danger'], row=1, col=2)
    
    fig.update_layout(
        title={'text': '📉 Residual Analysis', 'font': {'size': 20}},
        height=400,
        showlegend=False,
        **PLOTLY_TEMPLATE['layout']
    )
    
    return fig

def create_confusion_matrix_plot(y_test, y_pred):
    """Create premium confusion matrix."""
    y_test_class = y_test.apply(classify_happiness)
    y_pred_class = pd.Series(y_pred).apply(classify_happiness)
    
    labels = ['Low', 'Medium', 'High']
    cm = confusion_matrix(y_test_class, y_pred_class, labels=labels)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, '#1a1a2e'], [1, COLORS['accent']]],
        text=cm,
        texttemplate='%{text}',
        textfont={'size': 20, 'color': 'white'},
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': '🎲 Confusion Matrix', 'font': {'size': 20}},
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400,
        **PLOTLY_TEMPLATE['layout']
    )
    
    return fig

def create_year_distribution(df):
    """Create year distribution chart."""
    if 'year' not in df.columns:
        return None
    
    year_counts = df['year'].value_counts().sort_index()
    
    fig = go.Figure(go.Bar(
        x=year_counts.index.astype(int),
        y=year_counts.values,
        marker=dict(
            color=year_counts.values,
            colorscale=[[0, COLORS['primary']], [1, COLORS['accent']]],
            line=dict(color='white', width=1)
        ),
        hovertemplate='Year: %{x}<br>Records: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': '📅 Data by Year', 'font': {'size': 20}},
        xaxis_title='Year',
        yaxis_title='Number of Records',
        height=350,
        **PLOTLY_TEMPLATE['layout']
    )
    
    return fig

# ============================================================================
#                           MAIN APPLICATION
# ============================================================================

def main():
    # Hero Section
    st.markdown('<h1 class="hero-title">🌍 World Happiness Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">AI-Powered Happiness Score Prediction System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 Project Info")
        st.markdown("""
        <div class="premium-card">
            <p><strong>Student:</strong> Abdelrahman Mahmoud Seada</p>
            <p><strong>ID:</strong> 201801343</p>
            <p><strong>Instructor:</strong> Lamiaa Khairy</p>
            <p><strong>Course:</strong> Artificial Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎛️ Navigation")
        page = st.radio(
            "Select Section",
            ["📊 Dashboard", "🔮 Predict", "📈 Analysis", "📋 Data Explorer"],
            label_visibility="collapsed"
        )
    
    # Load data
    try:
        df = load_data()
        X, y, df_clean = prepare_data(df)
        model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(X, y)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # ========== DASHBOARD PAGE ==========
    if page == "📊 Dashboard":
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{metrics['test_r2']:.2%}", delta=f"{(metrics['test_r2']-0.7)*100:.1f}% above baseline")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.3f}", delta=f"-{metrics['rmse']:.3f} error")
        with col3:
            st.metric("MAE", f"{metrics['mae']:.3f}")
        with col4:
            st.metric("CV Score", f"{metrics['cv_mean']:.2%}", delta=f"±{metrics['cv_std']:.2%}")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_actual_vs_predicted(y_test, y_pred, metrics['test_r2']), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_feature_importance(model, list(X.columns)), use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_correlation_heatmap(X, y), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_confusion_matrix_plot(y_test, y_pred), use_container_width=True)
    
    # ========== PREDICT PAGE ==========
    elif page == "🔮 Predict":
        st.markdown("### 🔮 Predict Happiness Score")
        st.markdown("Adjust the sliders to predict a country's happiness score")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gdp = st.slider("💰 Economy (GDP per Capita)", 0.0, 2.0, 1.0, 0.01)
            family = st.slider("👨‍👩‍👧 Family Support", 0.0, 2.0, 1.0, 0.01)
            health = st.slider("🏥 Health (Life Expectancy)", 0.0, 2.0, 0.8, 0.01)
        
        with col2:
            freedom = st.slider("🗽 Freedom", 0.0, 1.0, 0.5, 0.01)
            trust = st.slider("🏛️ Trust (Low Corruption)", 0.0, 0.6, 0.2, 0.01)
            generosity = st.slider("🎁 Generosity", 0.0, 0.8, 0.2, 0.01)
        
        if st.button("🎯 Predict Happiness", use_container_width=True):
            input_data = pd.DataFrame({
                'Economy (GDP per Capita)': [gdp],
                'Family': [family],
                'Health (Life Expectancy)': [health],
                'Freedom': [freedom],
                'Trust (Government Corruption)': [trust],
                'Generosity': [generosity]
            })
            
            prediction = model.predict(input_data)[0]
            category = classify_happiness(prediction)
            
            color = COLORS['success'] if category == 'High' else (COLORS['warning'] if category == 'Medium' else COLORS['danger'])
            
            st.markdown(f"""
            <div class="premium-card" style="text-align: center;">
                <h2 style="color: {color}; font-size: 4rem; margin: 0;">{prediction:.2f}</h2>
                <p style="font-size: 1.5rem; color: #a0a0a0;">Predicted Happiness Score</p>
                <span class="success-badge" style="background: {color};">{category} Happiness</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation
            if prediction >= 7:
                st.success("🎉 Excellent! This would be among the happiest countries!")
            elif prediction >= 5.5:
                st.info("😊 Good! This indicates moderate to high happiness.")
            elif prediction >= 4:
                st.warning("😐 Room for improvement in happiness factors.")
            else:
                st.error("😟 This suggests many challenges to overcome.")
    
    # ========== ANALYSIS PAGE ==========
    elif page == "📈 Analysis":
        st.markdown("### 📈 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Residuals", "Year Distribution", "Feature Stats"])
        
        with tab1:
            st.plotly_chart(create_residual_plots(y_test, y_pred), use_container_width=True)
        
        with tab2:
            year_fig = create_year_distribution(df_clean)
            if year_fig:
                st.plotly_chart(year_fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### Feature Statistics")
            stats_df = X.describe().T
            stats_df['Coefficient'] = model.coef_
            st.dataframe(stats_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # ========== DATA EXPLORER PAGE ==========
    elif page == "📋 Data Explorer":
        st.markdown("### 📋 Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Valid Records", len(df_clean))
        with col3:
            st.metric("Features", len(Config.FEATURES))
        
        st.markdown("---")
        
        # Data preview
        st.markdown("#### 📊 Dataset Preview")
        st.dataframe(df_clean.head(50), use_container_width=True)
        
        # Download button
        csv = df_clean.to_csv(index=False)
        st.download_button(
            "📥 Download Clean Data",
            csv,
            "happiness_data_clean.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Made with ❤️ by <strong>Abdelrahman Mahmoud Seada</strong> (201801343)</p>
        <p>Instructor: Lamiaa Khairy | Course: Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
