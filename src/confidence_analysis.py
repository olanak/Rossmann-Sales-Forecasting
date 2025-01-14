# File: src/confidence_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging
from scipy.stats import norm

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load preprocessed validation data."""
    logger.info("Loading data...")
    data = pd.read_csv(data_path)
    logger.info(f"Data shape: {data.shape}")
    return data

def load_model(model_path):
    """Load the trained Random Forest model."""
    logger.info("Loading trained model...")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
    return model

def calculate_confidence_intervals(model, X, confidence_level=0.95):
    """Calculate confidence intervals for predictions."""
    logger.info("Calculating confidence intervals...")
    
    # Get predictions from each tree in the forest
    all_tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])

    # Mean and standard deviation of predictions
    mean_predictions = np.mean(all_tree_predictions, axis=0)
    std_predictions = np.std(all_tree_predictions, axis=0)

    # Z-score for the given confidence level
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate confidence intervals
    lower_bounds = mean_predictions - z_score * std_predictions
    upper_bounds = mean_predictions + z_score * std_predictions

    return mean_predictions, lower_bounds, upper_bounds

def plot_predictions_with_intervals(y_true, y_pred, lower_bounds, upper_bounds):
    """Plot actual vs predicted with confidence intervals."""
    logger.info("Plotting predictions with confidence intervals...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.values, label="Actual Sales", alpha=0.7)
    plt.plot(y_pred, label="Predicted Sales", alpha=0.7)
    plt.fill_between(range(len(y_pred)), lower_bounds, upper_bounds, color='gray', alpha=0.3, label="Confidence Interval")
    
    plt.xlabel("Samples")
    plt.ylabel("Sales")
    plt.title("Predictions with Confidence Intervals")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # File paths
    data_path = 'data/preprocessed_train.csv'
    model_path = 'models/random_forest_model.pkl'

    # Load data and model
    data = load_data(data_path)
    model = load_model(model_path)

    # Separate features and target
    feature_columns = [col for col in data.columns if col != 'Sales']
    X = data[feature_columns]
    y_true = data['Sales']

    # Calculate confidence intervals
    y_pred, lower_bounds, upper_bounds = calculate_confidence_intervals(model, X)

    # Plot predictions with confidence intervals
    plot_predictions_with_intervals(y_true, y_pred, lower_bounds, upper_bounds)
