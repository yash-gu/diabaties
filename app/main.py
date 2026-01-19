from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model at startup
model = joblib.load("app/model.pkl")

app = FastAPI(title="Diabetes Prediction API")


# Request schema: JSON body
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    # Order must match training features
    features = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age,
    ]])

    pred = model.predict(features)[0]
    proba = float(model.predict_proba(features)[0][1])

    return {
        "prediction": int(pred),
        "probability": proba
    }

