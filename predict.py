import pandas as pd
import numpy as np
import joblib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from tensorflow import keras

# Define the expected input data structure using Pydantic
# This still accepts the 7 features from your UI
class PredictionInput(BaseModel):
    size: float
    age: str
    windowType: str
    wallType: str
    floorType: str
    roofType: str
    propertyType: str  # This will be the raw string, e.g., "Semi-Detached"

# Initialize the FastAPI app
app = FastAPI()

# Allow the Vercel frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting to your Vercel domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load The Models (Preprocessor + NN Model) ---
base_path = os.path.dirname(__file__)
PREPROCESSOR_FILE = 'preprocessor.joblib'
MODEL_FILE = 'heatloss_nn_model.keras'

preprocessor_path = os.path.join(base_path, PREPROCESSOR_FILE)
model_path = os.path.join(base_path, MODEL_FILE)

preprocessor = None
model = None

try:
    # Load the preprocessor (ColumnTransformer)
    preprocessor = joblib.load(preprocessor_path)
    print(f"Preprocessor loaded successfully from {PREPROCESSOR_FILE}")
    
    # Load the Keras model
    model = keras.models.load_model(model_path)
    print(f"Keras model loaded successfully from {MODEL_FILE}")
    
except FileNotFoundError as e:
    print(f"Error: Model file not found. {e}")
except Exception as e:
    print(f"Error loading models: {e}")

# --- API Endpoint (WITH PROPERTYTYPE MAPPING) ---
@app.post("/api/predict")
async def predict_heatloss(input_data: PredictionInput):
    # Check if *both* models are loaded
    if preprocessor is None or model is None:
        return {"success": False, "error": "Models could not be loaded. Check server logs."}

    try:
# 1. Convert the single input item into a dictionary
        data = input_data.model_dump()
        
        # --- 2. Manually Map propertyType ---
        # Apply the same logic we used in the notebook
        # If it's not 'Bungalow', treat it as 'Detached' for the model
        if data['propertyType'].lower() != 'bungalow':
            data['propertyType'] = 'Detached'
        else:
            data['propertyType'] = 'Bungalow'
        # --- End of new logic ---

        # 3. Convert the *mapped* dictionary into a 1-row DataFrame
        input_df = pd.DataFrame([data])
        
        # 4. Preprocess the data
        # The .transform() method runs scaling and one-hot encoding
        processed_data = preprocessor.transform(input_df)
        
        # 5. Ensure data is dense (Keras needs a dense numpy array)
        if hasattr(processed_data, "toarray"):
            processed_data_dense = processed_data.toarray()
        else:
            processed_data_dense = processed_data
        
        # 6. Make the prediction with the Keras model
        # Keras returns a 2D array, e.g., [[prediction]]
        prediction_array = model.predict(processed_data_dense)
        
        # 7. Extract the single scalar value
        predicted_heatloss = prediction_array[0][0]

        # 8. Return the successful response
        return {
            "success": True,
            "predicted_heatloss_w": round(float(predicted_heatloss), 2)
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"success": False, "error": str(e)}

# A simple root endpoint to confirm the API is running
@app.get("/api/predict")
async def get_status():
    return {
        "status": "API is running",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": model is not None
    }