import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="`use_label_encoder` is deprecated in 1.7.0.")

# === LOAD AND LABEL DATA ===
def load_data(file_paths):
    # Load and concatenate all CSV files
    dfs = [pd.read_csv(path) for path in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()  # Clean column names

    print("\n=== Total Labeled Falls ===")
    print(df['isFall'].value_counts())

    # Adjust fall labels to mark the first onGround=True tick after a fall label
    fall_indices = df.index[df['isFall'] == 1]
    df['isFall'] = 0  # Reset all
    for idx in fall_indices:
        for look_ahead in range(1, 20):  # Look ahead up to 20 ticks
            if idx + look_ahead < len(df) and df.loc[idx + look_ahead, 'onGround']:
                df.loc[idx + look_ahead, 'isFall'] = 1
                break

    print("\n=== Reassigned Labeled Falls (First Ground Contact After Fall) ===")
    print(df['isFall'].value_counts())

    return df


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


# === BALANCE FALL/NON-FALL DATA ===
def balance_data(df, ratio=20, random_state=42):
    falls = df[df['isFall'] == 1]
    non_falls = df[df['isFall'] == 0]
    sample_size = min(len(non_falls), len(falls) * ratio)

    non_falls_sampled = non_falls.sample(n=sample_size, random_state=random_state)
    balanced_df = pd.concat([falls, non_falls_sampled]).sample(frac=1, random_state=random_state)

    print("\n=== Total Balanced Labeled Falls ===")
    print(balanced_df['isFall'].value_counts(), "\n")

    return balanced_df


# === MAIN TRAINING FLOW ===
if __name__ == "__main__":
    # Load data files
    file_paths = [
        'data/fall_data_2025-05-04_17-40-08.csv',
        'data/fall_data_2025-05-04_18-08-17.csv',
        'data/fall_data_2025-05-04_22-25-05.csv'
    ]
    
    df = load_data(file_paths)
    df = engineer_features(df)
    df = balance_data(df, ratio=20)

    # Feature set to train on
    features = [
        'onGround',              # 310.1483 – Strongest indicator of landing
        'onGround_ratio',        # 10.85453 – Time recently spent grounded
        'y',                     # 5.225263 – Current Y position
        'recent_y_min',          # 2.887518 – Lowest Y in recent window
        'deltaY',                # 2.802523 – Current Y change
        'low_velocity_duration', # 2.039953 – Time spent moving slowly
        'direction_changed',     # 1.878181 – Sudden direction switch
        'velY_prev',              # 1.099517 – Previous tick's vertical velocity
        'y_diff_from_prev_fall'
    ]

    X = df[features]
    y = df['isFall']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = XGBClassifier(scale_pos_weight=5, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict with confidence threshold
    threshold = 0.8
    probs = model.predict_proba(X_test)
    fall_probs = probs[:, 1]
    predictions = (fall_probs > threshold).astype(int)

    print(f"\n=== Threshold {threshold} ===")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, predictions))

    print("\n=== Prediction Counts ===")
    print(pd.Series(predictions).value_counts())

    # Save model
    joblib.dump(model, 'models/fall_model.joblib')

    # Feature importance (by gain)
    importance = model.get_booster().get_score(importance_type='gain')
    sorted_importance = pd.Series(importance).sort_values(ascending=False)
    print("\n=== Feature Importances (by Gain) ===")
    print(sorted_importance)