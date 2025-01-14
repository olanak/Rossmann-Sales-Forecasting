# File: src/lstm_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper Functions
def check_stationarity(data):
    """Check stationarity using Augmented Dickey-Fuller test."""
    logger.info("Checking stationarity...")
    result = adfuller(data)
    logger.info(f"ADF Statistic: {result[0]:.4f}")
    logger.info(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        logger.info("Data is stationary.")
    else:
        logger.info("Data is not stationary. Differencing is needed.")

def difference_data(data):
    """Difference the data to make it stationary."""
    logger.info("Differencing data...")
    return data.diff().dropna()

def plot_acf_pacf(data, lags=40):
    """Plot ACF and PACF for the data."""
    logger.info("Plotting ACF and PACF...")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(data, lags=lags, ax=ax[0])
    plot_pacf(data, lags=lags, ax=ax[1])
    plt.show()

def create_sequences(data, window_size):
    """Create sliding window sequences for supervised learning."""
    logger.info("Creating sequences...")
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def scale_data(train, val):
    """Scale data to the range (-1, 1)."""
    logger.info("Scaling data...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train.reshape(-1, 1))
    val_scaled = scaler.transform(val.reshape(-1, 1))
    return train_scaled, val_scaled, scaler

def build_lstm_model(input_shape):
    """Build an LSTM regression model."""
    logger.info("Building LSTM model...")
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    logger.info("Model built successfully.")
    return model

if __name__ == "__main__":
    # Load data
    data_path = 'data/preprocessed_train.csv'
    logger.info("Loading data...")
    data = pd.read_csv(data_path)

    # Isolate time series (aggregate sales by date if needed)
    logger.info("Isolating time series data...")
    sales_data = data.groupby('Date')['Sales'].sum()

    # Check stationarity
    check_stationarity(sales_data)

    # Difference data if needed
    sales_diff = difference_data(sales_data)

    # Plot ACF and PACF
    plot_acf_pacf(sales_diff)

    # Create supervised learning sequences
    window_size = 30  # Adjust based on the problem
    X, y = create_sequences(sales_diff.values, window_size)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train.flatten(), X_val.flatten())
    X_train_scaled = X_train_scaled.reshape(-1, window_size, 1)
    X_val_scaled = X_val_scaled.reshape(-1, window_size, 1)

    # Build and train LSTM model
    model = build_lstm_model((window_size, 1))
    logger.info("Training LSTM model...")
    model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=20, batch_size=32)

    # Save the trained model
    logger.info("Saving LSTM model...")
    model.save('models/lstm_model.h5')
    logger.info("Model saved successfully.")
