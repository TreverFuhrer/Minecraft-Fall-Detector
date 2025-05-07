import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from core.features import engineer_features
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

    # Find onGround=True after labeled fall, then assign the label to 10 ticks earlier
    fall_indices = []
    for idx in df.index[df['isFall'] == 1]:
        for look_ahead in range(1, 20):
            if idx + look_ahead < len(df) and df.loc[idx + look_ahead, 'onGround']:
                shifted_idx = idx + look_ahead - 10
                if shifted_idx >= 0:
                    fall_indices.append(shifted_idx)
                break

    df['isFall'] = 0
    df.loc[fall_indices, 'isFall'] = 1

    print("\n=== Reassigned Labeled Falls (First Ground Contact After Fall) ===")
    print(df['isFall'].value_counts())

    return df


# === BALANCE FALL/NON-FALL DATA ===
def balance_data(df, ratio, random_state=42):
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
    df = balance_data(df, ratio=60)

    # Feature set to train on
    features = [
        'onGround',
        'onGround_ratio',
        'y',
        'recent_y_min',
        'deltaY',
        'low_velocity_duration',
        'direction_changed',
        'velY_prev',
        #'y_diff_from_prev_fall',
        'delta_posX',
        'delta_posZ',
        'dirX_sign',
        'dirZ_sign',
        'dirX_change',
        'dirZ_change',
        'speed',
        'low_velocity',
        'y_diff_from_5ago',
        'y_climb_after_drop',
        'deltaY_prev',
        #'y_prev_fall',
        'has_prev_fall',
        #'ticks_since_prev_fall'
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