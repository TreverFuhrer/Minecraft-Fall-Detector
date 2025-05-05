import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

from xgboost import XGBClassifier

# === LOAD DATA ===
def load_data(file_paths):
    dfs = [pd.read_csv(path) for path in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.columns = combined_df.columns.str.strip()  # Remove whitespace in column names

    print("\n=== Total Labeled Falls ===")
    print(combined_df['isFall'].value_counts())

    # Expand fall labels to surrounding window
    FALL_WINDOW = 10
    fall_indices = combined_df.index[combined_df['isFall'] == 1]
    for idx in fall_indices:
        start = max(0, idx - FALL_WINDOW)
        end = min(len(combined_df) - 1, idx + FALL_WINDOW)
        combined_df.loc[start:end, 'isFall'] = 1 

    print("\n=== New Total Labeled Falls ===")
    print(combined_df['isFall'].value_counts())

    return combined_df



# === FEATURE ENGINEERING ===
def engineer_features(df, window=10):
    # Y movement change from previous tick
    df['deltaY'] = df['y'].diff().fillna(0)

    # Change in X and Z position (horizontal movement)
    df['delta_posX'] = df['x'].diff().fillna(0)
    df['delta_posZ'] = df['z'].diff().fillna(0)

    # Sign of horizontal movement (1 = positive, -1 = negative, 0 = no movement)
    df['dirX_sign'] = np.sign(df['delta_posX'])
    df['dirZ_sign'] = np.sign(df['delta_posZ'])

    # Whether X or Z direction changed in the last window (indicates turning around or hesitation)
    df['dirX_change'] = df['dirX_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
    df['dirZ_change'] = df['dirZ_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
    
    # If direction changed in either X or Z
    df['direction_changed'] = ((df['dirX_change'] + df['dirZ_change']) > 0).astype(int)

    # Total movement speed (magnitude of velocity vector)
    df['speed'] = (df['velX']**2 + df['velY']**2 + df['velZ']**2)**0.5

    # Was the player almost standing still? (<0.03 speed)
    df['low_velocity'] = (df['speed'] < 0.03).astype(int)

    # How many of the last 10 ticks had low velocity (possibly retrying parkour)
    df['low_velocity_duration'] = df['low_velocity'].rolling(window).sum().fillna(0)

    # Lowest Y value in the last window (used to detect drops)
    df['recent_y_min'] = df['y'].rolling(window).min().fillna(df['y'])

    # Did Y increase after 5 ticks ago (player climbed back up after falling)?
    df['y_diff_from_5ago'] = df['y'] - df['y'].shift(5)
    df['y_climb_after_drop'] = ((df['y_diff_from_5ago'] > 1.0) & (df['deltaY'] > 0)).astype(int)

    # Percent of last window spent on the ground
    df['onGround'] = df['onGround'].astype(int)
    df['onGround_ratio'] = df['onGround'].rolling(window).mean().fillna(0)

    # Ensure labels are integer type
    df['isFall'] = df['isFall'].astype(int)

    return df


# === BALANCE DATA ===
def balance_data(df, ratio=20, random_state=42):
    # Sample non-falls to maintain a fall:non-fall ratio
    falls = df[df['isFall'] == 1]
    non_falls = df[df['isFall'] == 0]
    num_to_keep = min(len(non_falls), len(falls) * ratio)
    non_falls_sampled = non_falls.sample(n=num_to_keep, random_state=random_state)
    balanced_df = pd.concat([falls, non_falls_sampled]).sample(frac=1, random_state=random_state)
    print("\n=== Total Balanced Labeled Falls ===")
    print(balanced_df['isFall'].value_counts())
    return balanced_df


# === MAIN TRAINING FLOW ===
if __name__ == "__main__":
    file_paths = [
        'data/fall_data_2025-05-04_17-40-08.csv',
        'data/fall_data_2025-05-04_18-08-17.csv',
        'data/fall_data_2025-05-04_22-25-05.csv'
    ]

    df = load_data(file_paths)
    df = engineer_features(df)
    df = balance_data(df, ratio=20)

    # Model inputs (features)
    features = [
        'y',                   # Current Y position
        'deltaY',              # Vertical movement
        'velY',                # Vertical velocity
        'onGround',            # Is the player on the ground
        # 'direction_changed', # Optional: turning around
        # 'low_velocity_duration', # Optional: standing still
        'recent_y_min',        # Lowest Y in last few ticks
        'y_climb_after_drop',  # Recovered upward movement after drop
        'onGround_ratio'       # % time spent grounded
    ]

    X = df[features]
    y = df['isFall']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train model
    model = XGBClassifier(scale_pos_weight=20, use_label_encoder=False, eval_metric='logloss')
    #model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, predictions))

    # Save model
    joblib.dump(model, 'models/fall_model.joblib')

    # Print counts of predictions
    print("\n=== Prediction Counts ===")
    print(pd.Series(predictions).value_counts())
