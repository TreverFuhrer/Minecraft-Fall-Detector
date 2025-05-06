import pandas as pd
import joblib
from ml.train_model import engineer_features




# === MAIN TRAINING FLOW ===
def main():
    # Load model
    model = joblib.load('fall_model.joblib')

    # Suppose you have a rolling buffer of 10 ticks
    buffer = pd.DataFrame([...])  # your most recent 10 ticks

    features_df = engineer_features(buffer.copy())

    # Grab just the latest row
    X_live = features_df.iloc[[-1]][model_features]  # model_features = list you trained on

    # Predict
    fall_prob = model.predict_proba(X_live)[0, 1]
    if fall_prob > 0.8:
        print("⚠️ Fall Detected")