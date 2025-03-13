from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from src.preprocessing import preprocess_data
from src.utils import setup_logger
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


logger = setup_logger()

def train_model(X_train, y_train, model_path="../../models/sklearn/"):
    # Define pipeline
    preprocessor = preprocess_data()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train model
    logger.info("Training Random Forest Regressor...")
    pipeline.fit(X_train, y_train)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_filename = f"{model_path}rossmann-rf-{timestamp}.pkl"
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(pipeline, model_filename)
    logger.info(f"Model saved to {model_filename}")
    
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred, squared=False)
    
    logger.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

def plot_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps['regressor']
    importances = model.feature_importances_
    
    # Match feature names after preprocessing
    onehot_columns = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
    all_features = np.concatenate([feature_names[:4], onehot_columns])
    
    # Plot feature importance
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [all_features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('../visualization/feature_importance.png')
    plt.show()