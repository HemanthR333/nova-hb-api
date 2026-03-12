from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="NOVA HB API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_data = joblib.load("nova_mae_under1_model.pkl")
lgb_model  = model_data['lgb_model']
gb_model   = model_data['gb_model']
rf_model   = model_data['rf_model']
scaler     = model_data['scaler']
features   = model_data['features']
weights    = model_data['weights']

class InputData(BaseModel):
    Gender: int
    Pregnancy: int
    Age: float
    Melanin_Index: float
    Red_IR_ratio: float
    NIR_Red_ratio: float
    F1_F8_ratio: float
    F5_555nm: float
    F7_630nm: float
    F8_680nm: float
    NIR_910nm: float
    Red: float
    IR: float

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: InputData):
    # Map back to original feature names with spaces
    input_dict = {
        "Gender": data.Gender,
        "Pregnancy": data.Pregnancy,
        "Age": data.Age,
        "Melanin_Index": data.Melanin_Index,
        "Red_IR_ratio": data.Red_IR_ratio,
        "NIR_Red_ratio": data.NIR_Red_ratio,
        "F1_F8_ratio": data.F1_F8_ratio,
        "F5 (555nm)": data.F5_555nm,
        "F7 (630nm)": data.F7_630nm,
        "F8 (680nm)": data.F8_680nm,
        "NIR (910nm)": data.NIR_910nm,
        "Red": data.Red,
        "IR": data.IR,
    }

    df = pd.DataFrame([input_dict])[features]
    scaled = scaler.transform(df)

    # Weighted ensemble prediction
    pred_lgb = lgb_model.predict(scaled)[0]
    pred_gb  = gb_model.predict(scaled)[0]
    pred_rf  = rf_model.predict(scaled)[0]

    final = (weights[0]*pred_lgb + weights[1]*pred_gb + weights[2]*pred_rf)

    return {
        "hemoglobin": round(float(final), 2),
        "lgb_pred": round(float(pred_lgb), 2),
        "gb_pred":  round(float(pred_gb), 2),
        "rf_pred":  round(float(pred_rf), 2)
    }
