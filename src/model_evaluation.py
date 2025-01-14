# post_prediction_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(train_path, val_path):
    """Load training and validation datasets."""
    logger.info("Loading data...")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    logger.info(f"Training data shape: {train.shape}")
    logger.info(f"Validation data shape: {val.shape}")
    return train, val

def load_model(model_path):
    """Load the trained model."""
    logger.info("Loading trained model...")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
    return model

def evaluate_predictions(y_true, y_pred):
    """Evaluate prediction metrics."""
    logger.info("Evaluating predictions...")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return rmse, mae, r2

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models."""
    logger.info("Plotting feature importance...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()
    else:
        logger.warning("The model does not support feature importance extraction.")

def plot_predictions(y_true, y_pred):
    """Plot actual vs predicted values."""
    logger.info("Plotting predictions...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.show()

def plot_residuals(y_true, y_pred):
    """Plot residuals of predictions."""
    logger.info("Plotting residuals...")
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.title("Residuals Distribution")
    plt.show()

if __name__ == "__main__":
    # File paths
    val_data_path = './data/preprocessed_train.csv'
    model_path = './models/random_forest_model.pkl'

    # Load validation data and model
    data = pd.read_csv(val_data_path)
    model = load_model(model_path)

    # Separate features and target
    feature_columns = [col for col in data.columns if col != 'Sales']
    X_val = data[feature_columns]
    y_val = data['Sales']

    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_val)

    # Evaluate predictions
    evaluate_predictions(y_val, y_pred)

    # Plot feature importance
    plot_feature_importance(model, feature_columns)

    # Plot actual vs predicted sales
    plot_predictions(y_val, y_pred)

    # Plot residuals
    plot_residuals(y_val, y_pred)
