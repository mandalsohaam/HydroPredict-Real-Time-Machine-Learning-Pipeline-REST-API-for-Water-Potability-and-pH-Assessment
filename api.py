from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib

app = FastAPI(title="Aqua-Epoch Water Quality REST API", version="1.0.0")

# Load artifacts
try:
    imputer = joblib.load('imputer.joblib')
    scaler = joblib.load('scaler.joblib')
    clf = joblib.load('clf_model.joblib')
    reg = joblib.load('reg_model.joblib')
except Exception:
    pass

class WaterQualityInput(BaseModel):
    hardness: float = Field(..., example=204.89)
    solids: float = Field(..., example=20791.32)
    chloramines: float = Field(..., example=7.30)
    sulfate: float = Field(..., example=368.51)
    conductivity: float = Field(..., example=421.88)
    organic_carbon: float = Field(..., example=10.37)
    trihalomethanes: float = Field(..., example=66.42)
    turbidity: float = Field(..., example=3.96)

@app.get("/")
def health_check():
    return {"status": "online", "system": "Aqua-Epoch HydroPredict API"}

@app.post("/predict")
def predict_water_quality(data: WaterQualityInput):
    features = np.array([[
        data.hardness, data.solids, data.chloramines,
        data.sulfate, data.conductivity, data.organic_carbon,
        data.trihalomethanes, data.turbidity
    ]])
    
    feat_imputed = imputer.transform(features)
    feat_scaled = scaler.transform(feat_imputed)
    
    # Task A: Classification
    potability_pred = int(clf.predict(feat_scaled)[0])
    potability_prob = float(clf.predict_proba(feat_scaled)[0][1])
    
    # Task B: Regression
    predicted_ph = float(reg.predict(feat_scaled)[0])
    
    return {
        "status": "success",
        "predictions": {
            "potability": {
                "class": potability_pred,
                "label": "Safe (Potable)" if potability_pred == 1 else "Unsafe (Non-Potable)",
                "confidence_score": round(potability_prob, 4)
            },
            "ph_level": {
                "predicted_ph": round(predicted_ph, 2),
                "evaluation": "Optimal WHO Standard" if 6.5 <= predicted_ph <= 8.5 else "Sub-optimal pH Level"
            }
        }
    }