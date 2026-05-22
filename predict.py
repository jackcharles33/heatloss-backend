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

        # Extract depth hint from wallType string sent by the frontend.
        # Frontend keys use patterns like:
        #   "cavity-post60-290-310-filled"   → BETWEEN_290_310
        #   "cavity-post60-under290-filled"  → LT_290
        #   "cavity-post60-310"              → GT_290  (310mm = wider filled cavity)
        # This was previously always None, stripping all depth signal from cavity
        # walls and causing solid walls to fall into the wrong U-value bucket.
        wall_raw = str(data.get('wallType', '')).lower()
        if 'gt_290' in wall_raw or 'gt290' in wall_raw:
            depth_hint = 'WALLS_DEPTH_GT_290'
        elif 'post60-310' in wall_raw:
            # "cavity-post60-310" key = 310mm wide cavity, wider than standard
            depth_hint = 'WALLS_DEPTH_GT_290'
        elif 'under290' in wall_raw or 'lt_290' in wall_raw or 'lt290' in wall_raw:
            depth_hint = 'WALLS_DEPTH_LT_290'
        elif '290_310' in wall_raw or '290-310' in wall_raw:
            depth_hint = 'WALLS_DEPTH_BETWEEN_290_310'
        else:
            depth_hint = None  # model will use safe numeric default (228mm)

        input_df = pd.DataFrame([{
            'ashp_survey_total_floor_area_sqm': data['size'],
            'property_age': data['age'],
            'walls_construction_type': data['wallType'],
            'windows_glazing': data['windowType'],
            'roof_type': data['roofType'],
            'property_floor_type': data['floorType'],
            'final_walls_depth': depth_hint,
            'roof_insulation_thickness': None,
            'final_floor_insulation_type': None,
            'walls_insulation': None
        }])

        preds = model.predict(input_df)

        heatloss_w   = float(preds['predicted_heatloss'].iloc[0])
        risk_flag    = bool(preds['is_unserviceable_risk'].iloc[0])
        borderline   = bool(preds['is_borderline'].iloc[0])
        safety_est   = float(preds['safety_estimate'].iloc[0])

        # ── Confidence score & interval ─────────────────────────────────────────
        # Matches the scoring in server.py / api/predict.js.
        # Base 72 %; era/size/wall bonuses; cap 88 %.
        wall_type = data.get('wallType', '')
        age       = data.get('age', '')
        size      = data.get('size', 100)

        confidence = 72

        if   age == 'POST_2008':          confidence += 10
        elif age == 'BETWEEN_2000_2008':  confidence += 7
        elif age == 'BETWEEN_1960_2000':  confidence += 4
        elif age == 'PRE_1960':           confidence += 2

        if   60  <= size <= 150: confidence += 5
        elif 150 <  size <= 220: confidence += 3
        elif 220 <  size <= 300: confidence += 1

        high_cavity = ('cavity-post60-310', 'cavity-post60-290-310-filled', 'cavity-post60-under290-filled')
        mid_cavity  = ('timber-frame', 'cavity-post60-290-310-unfilled', 'cavity-post60-under290-unfilled')
        pre60       = ('cavity-pre60-filled', 'cavity-pre60-unfilled')
        solid       = ('solid-brick-102', 'solid-brick-228', 'solid-brick-343')

        if   wall_type in high_cavity: confidence += 4
        elif wall_type in mid_cavity:  confidence += 3
        elif wall_type in pre60:       confidence += 2
        elif wall_type in solid:       confidence += 1

        confidence = min(88, confidence)

        margin_frac = (100 - confidence) * 0.013
        lower_bound = int(round(heatloss_w * (1 - margin_frac)))
        upper_bound = int(round(heatloss_w * (1 + margin_frac * 1.1)))

        return {
            "success": True,
            "predicted_heatloss_w":  round(heatloss_w, 0),
            "safety_estimate_w":     round(safety_est, 0),
            "is_unserviceable_risk": risk_flag,
            "is_borderline":         borderline,
            "confidence_score":      confidence,
            "lower_bound_w":         lower_bound,
            "upper_bound_w":         upper_bound,
            "model_info": "Physics-Hybrid-V3"
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