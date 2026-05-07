import pandas as pd
import numpy as np
import joblib
import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# Import the custom class so joblib can unpickle it
from production_model import HeatlossProductionModel

# Define Input Schema (Matches Frontend)
class PredictionInput(BaseModel):
    size: float
    age: str
    windowType: str
    wallType: str
    floorType: str
    roofType: str
    propertyType: str

# Init App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
base_path = os.path.dirname(__file__)
MODEL_FILE = 'production_model.joblib'
model_path = os.path.join(base_path, MODEL_FILE)
model = None
model_load_error = None

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Production model loaded successfully from {MODEL_FILE}")
    else:
        model_load_error = f"{MODEL_FILE} not found in {base_path}"
        print(f"Error: {model_load_error}")
except Exception as e:
    model_load_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    print(f"Error loading model:\n{model_load_error}")

@app.post("/api/predict")
async def predict_heatloss(input_data: PredictionInput):
    if model is None:
        return {"success": False, "error": "Model not loaded on server.", "detail": model_load_error}

    try:
        data = input_data.model_dump()

        input_df = pd.DataFrame([{
            'ashp_survey_total_floor_area_sqm': data['size'],
            'property_age': data['age'],
            'walls_construction_type': data['wallType'],
            'windows_glazing': data['windowType'],
            'roof_type': data['roofType'],
            'property_floor_type': data['floorType'],
            'final_walls_depth': None,
            'roof_insulation_thickness': None,
            'final_floor_insulation_type': None,
            'walls_insulation': None
        }])

        preds = model.predict(input_df)

        heatloss_w = float(preds['predicted_heatloss'].iloc[0])
        risk_flag = bool(preds['is_unserviceable_risk'].iloc[0])
        safety_est = float(preds['safety_estimate'].iloc[0])

        return {
            "success": True,
            "predicted_heatloss_w": round(heatloss_w, 0),
            "is_unserviceable_risk": risk_flag,
            "safety_estimate_w": round(safety_est, 0),
            "model_info": "Physics-Hybrid-V2"
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/predict")
async def get_status():
    return {
        "status": "API is running",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "type": "Physics-Hybrid"
    }