# File: api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize FastAPI App
app = FastAPI()

# Load the serialized model
MODEL_PATH = "../models/sklearn/rossmann-rf-latest.pkl"  # Update with your actual model path
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Define Input Schema
class InputData(BaseModel):
    StoreType: str
    Assortment: str
    Promo: int
    CompetitionDistance: float
    Promo2: int
    StateHoliday: str
    SchoolHoliday: int
    DayOfWeek: int
    Month: int
    Year: int
    WeekOfYear: int

# Define Prediction Endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    """
    Predict sales based on input features.
    """
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(data)
        
        # Return prediction
        return {"sales_prediction": round(prediction[0], 2)}
    
    except Exception as e:
        return {"error": str(e)}

# Health Check Endpoint
@app.get("/")
async def health_check():
    return {"status": "API is running"}