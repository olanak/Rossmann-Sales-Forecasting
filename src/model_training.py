import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(train_path):
    """Load preprocessed training data."""
    logger.info("Loading preprocessed training data...")
    data = pd.read_csv(train_path)
    logger.info(f"Loaded data shape: {data.shape}")
    return data

def split_data(data, target_column):
    """Split data into train and validation sets."""
    logger.info("Splitting data into train and validation sets...")
    
    # Select numeric columns and exclude non-numeric ones
    numeric_data = data.select_dtypes(include=[np.number])
    logger.info(f"Using numeric columns for training: {numeric_data.columns.tolist()}")
    
    # Define predictors and target variable
    X = numeric_data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val


def train_and_evaluate_models(X_train, X_val, y_train, y_val):
    """Train and evaluate multiple models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Create a pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_val)
        
        # Evaluate the model
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
        logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return results

if __name__ == "__main__":
    # Define file paths
    train_path = 'data/preprocessed_train.csv'
    
    # Load preprocessed data
    data = load_data(train_path)
    
    # Define target column
    target_column = 'Sales'
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(data, target_column)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_val, y_train, y_val)
    
    # Log results
    logger.info("Model evaluation results:")
    for model_name, metrics in results.items():
        logger.info(f"{model_name}: {metrics}")
    
    # Save the best model (Random Forest)
    best_model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    best_model_pipeline.fit(X_train, y_train)  # Refit the Random Forest pipeline on the entire training set
    import joblib
    joblib.dump(best_model_pipeline, 'models/random_forest_model.pkl')
    logger.info("Best model (Random Forest) saved successfully.")

