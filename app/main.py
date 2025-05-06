import pandas as pd
import numpy as np
import joblib
from ml.train_model import engineer_features

# Watchdog detects new tick
# Sends new tick to sliding buffer
# Buffer queues new tick, max queue size n

# === TICK BUFFER ===
def sliding_buffer(n=10):
    pass

# === FEATURE ENGINEERING ===
def engineer_features(df, window=10):
    # Movement deltas
    df['deltaY'] = df['y'].diff().fillna(0)
    df['delta_posX'] = df['x'].diff().fillna(0)
    df['delta_posZ'] = df['z'].diff().fillna(0)

    # Direction change detection
    df['dirX_sign'] = np.sign(df['delta_posX'])
    df['dirZ_sign'] = np.sign(df['delta_posZ'])
    df['dirX_change'] = df['dirX_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
    df['dirZ_change'] = df['dirZ_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
    df['direction_changed'] = ((df['dirX_change'] + df['dirZ_change']) > 0).astype(int)

    # Speed and low velocity features
    df['speed'] = np.sqrt(df['velX']**2 + df['velY']**2 + df['velZ']**2)
    df['low_velocity'] = (df['speed'] < 0.03).astype(int)
    df['low_velocity_duration'] = df['low_velocity'].rolling(window).sum().fillna(0)

    # Y position history
    df['recent_y_min'] = df['y'].rolling(window).min().fillna(df['y'])
    df['y_diff_from_5ago'] = df['y'] - df['y'].shift(5)
    df['y_climb_after_drop'] = ((df['y_diff_from_5ago'] > 1.0) & (df['deltaY'] > 0)).astype(int)

    # Ground-related features
    df['onGround'] = df['onGround'].astype(int)
    df['onGround_ratio'] = df['onGround'].rolling(window).mean().fillna(0)

    # Temporal velocity/Y features
    df['velY_prev'] = df['velY'].shift(1)
    df['deltaY_prev'] = df['y'] - df['y'].shift(1)

    df['y_prev_fall'] = np.nan
    last_y_fall = None
    for idx in df.index:
        if df.loc[idx, 'isFall'] == 1:
            if last_y_fall is not None:
                df.loc[idx, 'y_prev_fall'] = last_y_fall
            last_y_fall = df.loc[idx, 'y']
    df['y_diff_from_prev_fall'] = df['y'] - df['y_prev_fall']
    df['y_diff_from_prev_fall'] = df['y_diff_from_prev_fall'].fillna(0)



    df['isFall'] = df['isFall'].astype(int)

    return df

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