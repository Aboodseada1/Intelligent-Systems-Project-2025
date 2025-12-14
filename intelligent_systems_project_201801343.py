#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
    INTELLIGENT SYSTEMS PROJECT - WORLD HAPPINESS PREDICTION
=====================================================================
    
    Student Name: Abdelrahman Mahmoud Seada
    Student ID: 201801343
    Instructor: Lamiaa Khairy
    Course: Artificial Intelligence
    
    Description:
        This project analyzes the World Happiness Report dataset to 
        predict country happiness scores using Machine Learning.
        It implements Linear Regression with comprehensive data 
        preprocessing, visualization, and model evaluation.
    
    Dataset Source:
        World Happiness Ranking Dataset
        https://www.kaggle.com/datasets/nalisha/world-happiness-ranking-dataset
        
=====================================================================
"""

# ============================================================================
#                           IMPORT LIBRARIES
# ============================================================================

import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, List, Optional

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, UnicodeEncodeError):
        # Fallback: Use ASCII-safe output
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure matplotlib for interactive display
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
#                           CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for project parameters."""
    
    # File paths (relative to script location)
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DATASET_PATH = SCRIPT_DIR / "dataset" / "world_happiness_report.csv"
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    KNN_NEIGHBORS = 5
    CV_FOLDS = 5
    
    # Feature columns
    TARGET = 'Happiness Score'
    FEATURES = [
        'Economy (GDP per Capita)',
        'Family',
        'Health (Life Expectancy)',
        'Freedom',
        'Trust (Government Corruption)',
        'Generosity'
    ]
    
    # Happiness classification thresholds
    LOW_THRESHOLD = 4.5
    MEDIUM_THRESHOLD = 6.0


# ============================================================================
#                           DATA LOADING & PREPROCESSING
# ============================================================================

def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Load the World Happiness Report dataset from a local CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        ValueError: If the file is empty or invalid
    """
    print("\n" + "="*60)
    print("  STEP 1: DATA LOADING")
    print("="*60)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {filepath}\n"
            f"Please ensure the file exists in the 'dataset' folder."
        )
    
    df = pd.read_csv(filepath)
    
    if df.empty:
        raise ValueError("❌ The dataset file is empty!")
    
    print(f"✅ Successfully loaded dataset from:\n   {filepath}")
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns found:")
    for i, col in enumerate(df.columns.tolist(), 1):
        print(f"   {i:2d}. {col}")
    
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df: DataFrame to explore
    """
    print("\n" + "="*60)
    print("  STEP 2: DATA EXPLORATION")
    print("="*60)
    
    # Basic statistics
    print("\n📈 Summary Statistics:")
    print("-"*40)
    
    if Config.TARGET in df.columns:
        target_stats = df[Config.TARGET].describe()
        print(f"   • Mean Happiness Score: {target_stats['mean']:.3f}")
        print(f"   • Std Deviation: {target_stats['std']:.3f}")
        print(f"   • Min Score: {target_stats['min']:.3f}")
        print(f"   • Max Score: {target_stats['max']:.3f}")
    
    # Missing values summary
    print("\n🔍 Missing Values Summary:")
    print("-"*40)
    total_missing = df.isnull().sum().sum()
    print(f"   • Total missing values: {total_missing}")
    
    if total_missing > 0:
        missing_by_col = df.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"   • {col}: {count} ({pct:.1f}%)")
    
    # Year distribution (if available)
    if 'year' in df.columns:
        print("\n📅 Year Distribution:")
        print("-"*40)
        year_counts = df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"   • {int(year)}: {count} records")


def clean_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean the dataset and prepare features for modeling.
    
    This function:
    1. Drops rows where the target variable is missing
    2. Fills missing feature values using KNN imputation
    3. Prepares feature matrix X and target vector y
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    print("\n" + "="*60)
    print("  STEP 3: DATA CLEANING & PREPARATION")
    print("="*60)
    
    original_count = len(df)
    
    # Step 1: Check for target column
    if Config.TARGET not in df.columns:
        raise ValueError(f"❌ Target column '{Config.TARGET}' not found in dataset!")
    
    # Step 2: Drop rows with missing target values
    df_clean = df.dropna(subset=[Config.TARGET]).copy()
    dropped_count = original_count - len(df_clean)
    
    print(f"\n🧹 Cleaning Results:")
    print(f"   • Original rows: {original_count}")
    print(f"   • Rows with missing target: {dropped_count}")
    print(f"   • Remaining rows: {len(df_clean)}")
    
    # Step 3: Verify feature columns exist
    available_features = [f for f in Config.FEATURES if f in df_clean.columns]
    missing_features = [f for f in Config.FEATURES if f not in df_clean.columns]
    
    if missing_features:
        print(f"\n⚠️  Warning: Some features not found in dataset:")
        for f in missing_features:
            print(f"   - {f}")
        print(f"\n   Using available features: {len(available_features)}/{len(Config.FEATURES)}")
    
    # Step 4: Extract features and target
    X = df_clean[available_features].copy()
    y = df_clean[Config.TARGET].copy()
    
    # Step 5: Handle missing feature values with KNN imputation
    missing_features_count = X.isnull().sum().sum()
    
    if missing_features_count > 0:
        print(f"\n🔧 Imputing {missing_features_count} missing feature values using KNN...")
        imputer = KNNImputer(n_neighbors=Config.KNN_NEIGHBORS)
        X_filled = imputer.fit_transform(X)
        X = pd.DataFrame(X_filled, columns=available_features, index=X.index)
        print("   ✅ Imputation completed!")
    else:
        print("\n✅ No missing feature values - no imputation needed!")
    
    # Reset indices for consistency
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y


# ============================================================================
#                           VISUALIZATION FUNCTIONS
# ============================================================================

def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Create a correlation heatmap for features and target.
    
    Args:
        X: Feature DataFrame
        y: Target Series
    """
    print("\n📊 Generating correlation heatmap...")
    
    # Combine features and target for correlation analysis
    data = X.copy()
    data[Config.TARGET] = y
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='RdYlBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model: LinearRegression, feature_names: List[str]) -> None:
    """
    Create a horizontal bar chart showing feature importance.
    
    Args:
        model: Trained LinearRegression model
        feature_names: List of feature names
    """
    print("\n📊 Generating feature importance plot...")
    
    # Get coefficients and sort by absolute value
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    importance = importance.sort_values('Coefficient', key=abs, ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c' if x < 0 else '#27ae60' for x in importance['Coefficient']]
    
    plt.barh(importance['Feature'], importance['Coefficient'], color=colors, edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance (Linear Regression Coefficients)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray, r2: float) -> None:
    """
    Create scatter plot comparing actual vs predicted values.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        r2: R² score for annotation
    """
    print("\n📊 Generating actual vs predicted plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, c='#3498db', edgecolors='white', s=80)
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add R² annotation
    plt.annotate(
        f'R² = {r2:.4f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=14,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.xlabel('Actual Happiness Score', fontsize=12)
    plt.ylabel('Predicted Happiness Score', fontsize=12)
    plt.title('Actual vs Predicted Happiness Scores', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Create residual plot to assess model performance.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
    """
    print("\n📊 Generating residual plot...")
    
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, c='#9b59b6', edgecolors='white', s=60)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=30, color='#1abc9c', edgecolor='white', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Value', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
#                           MODEL TRAINING & EVALUATION
# ============================================================================

def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*60)
    print("  STEP 4: DATA SPLITTING")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    
    print(f"\n✂️  Data Split Results:")
    print(f"   • Training samples: {len(X_train)} ({(1-Config.TEST_SIZE)*100:.0f}%)")
    print(f"   • Testing samples: {len(X_test)} ({Config.TEST_SIZE*100:.0f}%)")
    print(f"   • Random state: {Config.RANDOM_STATE}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained LinearRegression model
    """
    print("\n" + "="*60)
    print("  STEP 5: MODEL TRAINING")
    print("="*60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("\n✅ Linear Regression model trained successfully!")
    print(f"\n📐 Model Parameters:")
    print(f"   • Intercept: {model.intercept_:.4f}")
    print(f"\n   Feature Coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        sign = '+' if coef >= 0 else ''
        print(f"   • {feature}: {sign}{coef:.4f}")
    
    return model


def evaluate_model(model: LinearRegression, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, dict]:
    """
    Evaluate the trained model using multiple metrics.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        
    Returns:
        Tuple of (predictions array, metrics dictionary)
    """
    print("\n" + "="*60)
    print("  STEP 6: MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Calculate regression metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS, scoring='r2')
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    print("\n📊 Model Performance Metrics:")
    print("-"*50)
    print(f"   Training R² Score:     {metrics['train_r2']:.4f}")
    print(f"   Testing R² Score:      {metrics['test_r2']:.4f}")
    print(f"   Mean Squared Error:    {metrics['mse']:.4f}")
    print(f"   Root Mean Sq Error:    {metrics['rmse']:.4f}")
    print(f"   Mean Absolute Error:   {metrics['mae']:.4f}")
    print(f"\n   Cross-Validation ({Config.CV_FOLDS}-fold):")
    print(f"   • Mean R² Score:       {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    
    # Check for overfitting
    overfit_gap = metrics['train_r2'] - metrics['test_r2']
    if overfit_gap > 0.1:
        print(f"\n⚠️  Warning: Possible overfitting detected!")
        print(f"   Train-Test R² gap: {overfit_gap:.4f}")
    else:
        print(f"\n✅ Model generalizes well (Train-Test gap: {overfit_gap:.4f})")
    
    return y_pred, metrics


def classify_happiness(score: float) -> str:
    """
    Classify happiness score into categories.
    
    Args:
        score: Happiness score value
        
    Returns:
        Category string ('Low', 'Medium', or 'High')
    """
    if score < Config.LOW_THRESHOLD:
        return 'Low'
    elif score < Config.MEDIUM_THRESHOLD:
        return 'Medium'
    else:
        return 'High'


def generate_classification_report(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Generate classification report and confusion matrix.
    
    This treats the regression problem as a classification task
    by binning scores into Low/Medium/High categories.
    
    Args:
        y_test: Actual happiness scores
        y_pred: Predicted happiness scores
    """
    print("\n" + "="*60)
    print("  STEP 7: CLASSIFICATION ANALYSIS")
    print("="*60)
    
    # Convert to categories
    y_test_class = y_test.apply(classify_happiness)
    y_pred_class = pd.Series(y_pred).apply(classify_happiness)
    
    labels = ['Low', 'Medium', 'High']
    
    print("\n🏷️  Happiness Score Categories:")
    print(f"   • Low: < {Config.LOW_THRESHOLD}")
    print(f"   • Medium: {Config.LOW_THRESHOLD} - {Config.MEDIUM_THRESHOLD}")
    print(f"   • High: > {Config.MEDIUM_THRESHOLD}")
    
    print("\n📋 Classification Report:")
    print("-"*50)
    print(classification_report(y_test_class, y_pred_class, labels=labels, zero_division=0))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_class, y_pred_class, labels=labels)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix (Happiness Categories)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
#                           INTERACTIVE PREDICTION
# ============================================================================

def interactive_prediction(model: LinearRegression, feature_names: List[str]) -> None:
    """
    Allow users to input values and get happiness predictions.
    
    Args:
        model: Trained model
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("  🔮 INTERACTIVE PREDICTION")
    print("="*60)
    print("\nEnter values for each feature (typically between 0.0 and 2.0)")
    print("Type 'quit' to exit.\n")
    
    feature_prompts = {
        'Economy (GDP per Capita)': 'Economy/GDP',
        'Family': 'Family Support',
        'Health (Life Expectancy)': 'Health/Life Expectancy',
        'Freedom': 'Freedom',
        'Trust (Government Corruption)': 'Trust/Low Corruption',
        'Generosity': 'Generosity'
    }
    
    while True:
        try:
            print("-"*40)
            values = []
            
            for feature in feature_names:
                prompt = feature_prompts.get(feature, feature)
                user_input = input(f"  {prompt}: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\n👋 Exiting interactive mode. Goodbye!")
                    return
                
                values.append(float(user_input))
            
            # Create DataFrame for prediction
            input_data = pd.DataFrame([values], columns=feature_names)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            category = classify_happiness(prediction)
            
            # Display result
            print("\n" + "="*40)
            print(f"  🎯 Predicted Happiness Score: {prediction:.2f}")
            print(f"  📊 Category: {category}")
            print("="*40)
            
            # Fun feedback
            if prediction >= 7:
                print("  🎉 Excellent! This would be a very happy country!")
            elif prediction >= 5.5:
                print("  😊 Good! This indicates moderate happiness.")
            elif prediction >= 4:
                print("  😐 Room for improvement in happiness factors.")
            else:
                print("  😟 This suggests many challenges to overcome.")
            
            print()
            
        except ValueError:
            print("\n❌ Invalid input. Please enter numeric values only.\n")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting interactive mode.")
            return


# ============================================================================
#                           MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete analysis pipeline."""
    
    # Print header
    print("\n" + "="*70)
    print("  " + "="*66)
    print("  ||     INTELLIGENT SYSTEMS PROJECT - HAPPINESS PREDICTION     ||")
    print("  ||                                                            ||")
    print("  ||     Student: Abdelrahman Mahmoud Seada                     ||")
    print("  ||     ID: 201801343                                          ||")
    print("  ||     Instructor: Lamiaa Khairy                              ||")
    print("  " + "="*66)
    print("="*70)
    
    try:
        # Step 1: Load data
        df = load_dataset(Config.DATASET_PATH)
        
        # Step 2: Explore data
        explore_data(df)
        
        # Step 3: Clean and prepare data
        X, y = clean_and_prepare_data(df)
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 5: Train model
        model = train_model(X_train, y_train)
        
        # Step 6: Evaluate model
        y_pred, metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Step 7: Generate visualizations
        print("\n" + "="*60)
        print("  STEP 8: VISUALIZATIONS")
        print("="*60)
        
        plot_correlation_heatmap(X, y)
        plot_feature_importance(model, list(X.columns))
        plot_actual_vs_predicted(y_test, y_pred, metrics['test_r2'])
        plot_residuals(y_test, y_pred)
        
        # Step 8: Classification analysis
        generate_classification_report(y_test, y_pred)
        
        # Step 9: Interactive prediction (optional)
        print("\n" + "="*60)
        print("  ANALYSIS COMPLETE!")
        print("="*60)
        
        try_interactive = input("\n🔮 Would you like to try interactive prediction? (yes/no): ").strip().lower()
        if try_interactive in ['yes', 'y']:
            interactive_prediction(model, list(X.columns))
        
        print("\n" + "="*60)
        print("  Thank you for using the Happiness Prediction System!")
        print("  Student: Abdelrahman Mahmoud Seada (201801343)")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()