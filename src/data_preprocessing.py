# File: src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(train_path, test_path, store_path):
    """Load datasets."""
    logger.info("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    store = pd.read_csv(store_path)
    logger.info(f"Train dataset shape: {train.shape}")
    logger.info(f"Test dataset shape: {test.shape}")
    logger.info(f"Store dataset shape: {store.shape}")
    return train, test, store

def handle_missing_values(train, test, store):
    """Handle missing values in datasets."""
    logger.info("Handling missing values...")
    store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].median())
    store.fillna(0, inplace=True)
    train = train.merge(store, on='Store', how='left')
    test = test.merge(store, on='Store', how='left')
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    logger.info("Missing values handled successfully.")
    return train, test

def encode_categorical(train, test):
    """Encode categorical variables."""
    logger.info("Encoding categorical variables...")
    le = LabelEncoder()
    for col in ['StateHoliday', 'StoreType', 'Assortment']:
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    logger.info("Categorical encoding completed.")
    return train, test

def scale_features(train, test, features):
    """Scale numeric features."""
    logger.info("Scaling numeric features...")

    # Identify common features available in both train and test datasets
    common_features = [feature for feature in features if feature in train.columns and feature in test.columns]
    
    # Log excluded features
    excluded_features = [feature for feature in features if feature not in common_features]
    if excluded_features:
        logger.warning(f"The following features were excluded from scaling as they are missing in one of the datasets: {excluded_features}")
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Scale only the common features
    train[common_features] = scaler.fit_transform(train[common_features])
    test[common_features] = scaler.transform(test[common_features])
    
    logger.info("Feature scaling completed.")
    return train, test



def generate_features(train, test):
    """Generate new features from date column."""
    logger.info("Generating new features...")
    for df in [train, test]:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    logger.info("Feature generation completed.")
    return train, test

if __name__ == "__main__":
    # Define file paths
    train_path = 'data/train.csv'  # Adjusted to correct location
    test_path = 'data/test.csv'
    store_path = 'data/store.csv'

    # Load datasets
    train, test, store = load_data(train_path, test_path, store_path)

    # Handle missing values
    train, test = handle_missing_values(train, test, store)

    # Encode categorical variables
    train, test = encode_categorical(train, test)

    # Scale numeric features
    numeric_features = ['Customers', 'CompetitionDistance']
    train, test = scale_features(train, test, numeric_features)

    # Generate new features
    train, test = generate_features(train, test)

 # Save preprocessed data
    train.to_csv('data/preprocessed_train.csv', index=False)
    test.to_csv('data/preprocessed_test.csv', index=False)
    logger.info("Preprocessed data saved successfully.")
