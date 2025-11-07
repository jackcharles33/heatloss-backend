import pandas as pd
import joblib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# Define the expected input data structure using Pydantic
# This matches the 6 features from your UI
class PredictionInput(BaseModel):
    size: float
    age: str
    windowType: str
    wallType: str
    floorType: str
    roofType: str

# Initialize the FastAPI app
app = FastAPI()

# Allow the Vercel frontend to call this API (configure origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting to your Vercel domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load The Model ---
# Construct the full path to the model file
# Vercel places API files in a specific directory, so we find it relative to this .py file
model_path = os.path.join(os.path.dirname(__file__), '5000Model.joblib')

model = None

try:
    # Load the model into memory when the API starts
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# --- API Endpoint ---
@app.post("/api/predict")
async def predict_heatloss(input_data: PredictionInput):
    if model is None:
        return {"success": False, "error": "Model could not be loaded. Check server logs."}

    try:
        # 1. Convert the single input item into a dictionary
        data = input_data.model_dump()
        
        # 2. Convert the dictionary into a 1-row pandas DataFrame
        # This is the exact format the scikit-learn pipeline expects
        input_df = pd.DataFrame([data])
        
        # 3. Make the prediction
        # The .predict() method runs the full pipeline:
        # (imputing, one-hot encoding, and predicting)
        prediction_array = model.predict(input_df)
        
        # 4. Extract the single prediction value
        predicted_heatloss = prediction_array[0]

        # 5. Return the successful response
        return {
            "success": True,
            "predicted_heatloss_w": round(predicted_heatloss, 2)
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"success": False, "error": str(e)}

# A simple root endpoint to confirm the API is running
@app.get("/api/predict")
async def get_status():
    return {"status": "API is running", "model_loaded": model is not None}