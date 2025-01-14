# Rossmann Sales Forecasting

This project predicts daily sales for Rossmann stores using machine learning and deep learning models. The application provides a REST API for real-time predictions.

---

Project Structure

```
Rossmann-Sales-Forecasting/
│
├── data/                  # Dataset and preprocessed files
│   ├── train.csv          # Raw training data
│   ├── test.csv           # Raw test data
│   ├── store.csv          # Store metadata
│   ├── preprocessed_train.csv  # Preprocessed training data
│
├── models/                # Trained model files
│   ├── random_forest_model.pkl
│   ├── lstm_model.h5
│
├── src/                   # Source code for the project
│   ├── data_preprocessing.py   # Preprocessing script
│   ├── model_training.py       # Training script for ML models
│   ├── lstm_model.py           # LSTM model training
│   ├── serve_model.py          # REST API implementation
│   ├── post_prediction_analysis.py  # Post-prediction analysis
│
└── README.md             # Project documentation
```

---

Getting Started

1. Clone the Repository
```
git clone https://github.com/<your-username>/Rossmann-Sales-Forecasting.git
cd Rossmann-Sales-Forecasting
```

2. Set Up the Environment
- Install Python 3.8 or higher.
- Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- Install dependencies:
```
pip install -r requirements.txt
```

3. Data Preprocessing
- Preprocess the data for training:
```bash
python3 src/data_preprocessing.py
```

4. Train the Models
- Train the machine learning models:
```
python3 src/model_training.py
```
- Train the LSTM model:
```
python3 src/lstm_model.py
```

5. Start the REST API
- Serve the models through a REST API:
```
python3 src/serve_model.py
```
- Test the API endpoints:
  - Random Forest:
    ```
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Store": 1, "DayOfWeek": 5, "Promo": 1, "StateHoliday": 0, "SchoolHoliday": 0, "Customers": 200, "CompetitionDistance": 500}'
    ```
  - LSTM:
    ```
    curl -X POST http://127.0.0.1:5000/predict-lstm -H "Content-Type: application/json" -d '{"sequence": [0.1, 0.2, 0.3, 0.4, 0.5]}'
    ```

---

## Features

- Data Preprocessing: Handle missing values, encode categorical variables, and scale numeric features.
- Machine Learning: Random Forest and Gradient Boosting models for sales prediction.
- Deep Learning: LSTM model for time series sales forecasting.
- REST API: Real-time predictions using the trained models.

---

## Dependencies

- Python 3.8+
- Flask
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

Install all dependencies using:
```
pip install -r requirements.txt
``

---

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.

---
## Contact
For any questions or support, please contact:
- Name: Olana Kenea
- Email: olanakenea6@gmail.com

