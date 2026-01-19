from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load model from app folder (where main.py is)
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print("Model loaded successfully!")

app = FastAPI(title="Diabetes Prediction API")

class DiabetesData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
async def root():
    """Serve the HTML frontend"""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    return FileResponse(template_path, media_type='text/html')

@app.post("/predict")
async def predict(data: DiabetesData):
    """Predict diabetes risk"""
    input_array = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])
    
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][int(prediction)]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk": "High Risk" if prediction == 1 else "Low Risk"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
