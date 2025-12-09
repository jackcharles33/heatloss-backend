import pandas as pd
import joblib
from production_model import HeatlossProductionModel

# Config
DATA_FILE = 'Survey_Data_for_HL (10).csv'
TARGET = 'ashp_survey_total_property_heatloss_w'
OUTPUT_FILE = 'production_model.joblib'

def train_and_save():
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE).dropna(subset=[TARGET])
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    print(f"Training HeatlossProductionModel on {len(df)} records...")
    model = HeatlossProductionModel(random_state=42)
    model.fit(X, y)
    
    print(f"Saving model to {OUTPUT_FILE}...")
    joblib.dump(model, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    train_and_save()
