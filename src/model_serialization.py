# File: src/model_serialization.py

import joblib
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_serialize_model(train_path, model_path):
    """Train a Random Forest model and serialize it."""
    logger.info("Loading preprocessed training data...")
    data = pd.read_csv(train_path)

    # Define features and target
    feature_columns = [col for col in data.columns if col != 'Sales']
    X = data[feature_columns]
    y = data['Sales']

    logger.info("Training Random Forest model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    logger.info("Model training completed.")

    # Serialize the model
    logger.info(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    logger.info("Model saved successfully.")

def load_model(model_path):
    """Load a serialized model."""
    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
    return model

if __name__ == "__main__":
    # File paths
    train_data_path = 'data/preprocessed_train.csv'
    model_save_path = 'models/random_forest_model.pkl'

    # Train and serialize the model
    train_and_serialize_model(train_data_path, model_save_path)

    # Test loading the model
    model = load_model(model_save_path)
    logger.info("Model ready for predictions.")
