# **Rossmann Store Sales Forecasting**

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue) ![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/rossmann-sales-forecasting)

## **Overview**
This repository contains an end-to-end solution for forecasting daily sales in Rossmann stores across several cities up to six weeks in advance. The project leverages advanced machine learning and deep learning techniques to predict sales, enabling the finance team to make data-driven decisions regarding resource allocation, promotions, and operational planning.

The solution is modular, scalable, and production-ready, with a REST API for real-time predictions, comprehensive exploratory data analysis (EDA), and robust model management practices.

---

## **Table of Contents**
1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Documentation](#api-documentation)
6. [Contributing](#contributing)
7. [License](#license)
8. [References](#references)

---

## **Project Structure**
```
rossmann-sales-forecasting/
├── api/                      # API code for serving predictions
│   ├── app.py                # FastAPI application
│   └── requirements.txt      # Dependencies for the API
├── data/
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data (optional)
├── models/                   # Serialized models
│   └── sklearn/              # Scikit-learn models
├── notebooks/
│   ├── exploration/
│   │   └── 01_eda_customer_behavior.ipynb  # Exploratory Data Analysis
│   ├── modeling/
│   │   ├── 02_sales_prediction_sklearn.ipynb  # Machine Learning Modeling
│   │   └── 03_sales_prediction_lstm.ipynb     # Deep Learning Modeling
├── reports/                  # Analysis results and visualizations
│   ├── figures/              # Visualization outputs
│   └── performance/          # Model evaluation metrics
├── src/                      # Python modules for preprocessing, modeling, etc.
└── README.md                 # This file
```

---

## **Features**
### **1. Exploratory Data Analysis (EDA)**
- Comprehensive cleaning and preprocessing pipelines.
- Feature engineering to extract insights from temporal and categorical data.
- Advanced visualizations using `matplotlib`, `seaborn`, and `plotly`.

### **2. Machine Learning Modeling**
- Preprocessing pipeline with `sklearn` for handling missing values, scaling, and encoding.
- Trained models include:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost
- Post-prediction analysis with SHAP for feature importance and confidence intervals.

### **3. Deep Learning Modeling**
- LSTM-based time series forecasting using TensorFlow or PyTorch.
- Sliding window technique for converting time series data into supervised learning format.
- Evaluation metrics like SMAPE for interpretable performance.

### **4. Model Serving API**
- REST API built with FastAPI for real-time predictions.
- Input validation using Pydantic models.
- Containerized deployment with Docker for scalability.

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/olanak/rossmann-sales-forecasting.git
cd rossmann-sales-forecasting
```

### **2. Install Dependencies**
#### **For Jupyter Notebooks:**
```bash
pip install -r requirements.txt
```

#### **For the API:**
Navigate to the `api/` directory and install dependencies:
```bash
cd api/
pip install -r requirements.txt
```

### **3. Download Dataset**
Download the dataset from [Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) and place it in the `data/raw/` directory.

---

## **Usage**
### **1. Run Exploratory Data Analysis**
Open and execute the notebook `notebooks/exploration/01_eda_customer_behavior.ipynb` to explore customer purchasing behavior and derive actionable insights.

### **2. Train Machine Learning Models**
Run the notebook `notebooks/modeling/02_sales_prediction_sklearn.ipynb` to preprocess data, train models, and evaluate performance.

### **3. Train Deep Learning Models**
Execute the notebook `notebooks/modeling/03_sales_prediction_lstm.ipynb` to train an LSTM model for time series forecasting.

### **4. Serve Predictions via API**
Start the FastAPI server:
```bash
cd api/
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

#### **Test the API:**
Use `curl` or Postman to send a POST request to `/predict`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "StoreType": "a",
    "Assortment": "basic",
    "Promo": 1,
    "CompetitionDistance": 5000,
    "StateHoliday": "0",
    "SchoolHoliday": 0,
    "DayOfWeek": 3,
    "Month": 10,
    "Year": 2023,
    "WeekOfYear": 40
}'
```

---

## **API Documentation**
Access the interactive API documentation at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## **Contributing**
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m "Add YourFeatureName"`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **References**
1. Kaggle Dataset: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
2. FastAPI Documentation: [FastAPI](https://fastapi.tiangolo.com/)
3. SHAP Library: [SHAP](https://shap.readthedocs.io/en/latest/)
4. Time Series Forecasting with LSTM: [LSTM Tutorial](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
5. Sklearn Pipelines: [Sklearn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

---

## **Acknowledgments**
Special thanks to the 10 Academy team for providing guidance and support throughout this project. Additional thanks to the Kaggle community for sharing valuable kernels and insights.

---

Feel free to reach out with any questions or suggestions! 🚀
