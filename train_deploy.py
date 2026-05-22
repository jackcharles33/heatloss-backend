import pandas as pd
import joblib
from production_model import HeatlossProductionModel
from preprocess import preprocess, TARGET

# Config
DATA_FILE   = 'heatlossdata2.csv'
OUTPUT_FILE = 'production_model.joblib'

def train_and_save():
    print(f"Loading data from {DATA_FILE}...")
    raw = pd.read_csv(DATA_FILE)
    print(f"  {len(raw):,} rows loaded")

    df = preprocess(raw)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"Training HeatlossProductionModel on {len(df):,} records...")
    model = HeatlossProductionModel(random_state=42)
    model.fit(X, y)

    print(f"Saving model to {OUTPUT_FILE}...")
    joblib.dump(model, OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    train_and_save()
