# File: src/serve_model.py

from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import logging

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load models
MODEL_PATH_RF = 'models/random_forest_model.pkl'
MODEL_PATH_LSTM = 'models/lstm_model.h5'
logger.info("Loading trained models...")
random_forest_model = joblib.load(MODEL_PATH_RF)
lstm_model = tf.keras.models.load_model(MODEL_PATH_LSTM)
logger.info("Models loaded successfully.")

@app.route('/')
def home():
    return "Model Serving API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for Random Forest predictions."""
    try:
        # Parse input JSON
        input_data = request.get_json()
        logger.info(f"Received input data: {input_data}")

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions with Random Forest
        rf_predictions = random_forest_model.predict(input_df)

        # Return predictions as JSON
        return jsonify({"random_forest_predictions": rf_predictions.tolist()})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict-lstm', methods=['POST'])
def predict_lstm():
    """Endpoint for LSTM predictions."""
    try:
        # Parse input JSON
        input_data = request.get_json()
        logger.info(f"Received input data: {input_data}")

        # Convert JSON to Numpy Array for LSTM
        input_array = np.array(input_data['sequence']).reshape(1, -1, 1)

        # Make predictions with LSTM
        lstm_predictions = lstm_model.predict(input_array)

        # Return predictions as JSON
        return jsonify({"lstm_predictions": lstm_predictions.flatten().tolist()})

    except Exception as e:
        logger.error(f"Error during LSTM prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
