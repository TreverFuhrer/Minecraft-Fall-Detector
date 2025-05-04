import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Read in data from csv
df1 = pd.read_csv('data/Fall Data May 4 2025.csv')
df = pd.concat([df1], ignore_index=True)

print(df.head())
print("--------------------------")


window = 10
# Features

# Add a delta features
df['deltaY'] = df['y'].diff().fillna(0)
df['delta_posX'] = df['x'].diff().fillna(0)
df['delta_posZ'] = df['z'].diff().fillna(0)

# Sign of movement (positive = right/forward, negative = left/back)
df['dirX_sign'] = np.sign(df['delta_posX'])
df['dirZ_sign'] = np.sign(df['delta_posZ'])

# Direction changed feature
df['dirX_change'] = df['dirX_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
df['dirZ_change'] = df['dirZ_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
df['direction_changed'] = ((df['dirX_change'] + df['dirZ_change']) > 0).astype(int)

# Count how many ticks out of the last 10 had low velocity/ slow (almost stationary)
df['speed'] = (df['velX']**2 + df['velY']**2 + df['velZ']**2)**0.5
df['low_velocity'] = (df['speed'] < 0.03).astype(int)  # You can tweak the threshold
df['low_velocity_duration'] = df['low_velocity'].rolling(window).sum().fillna(0)

# The lowest Y-value in the last window
df['recent_y_min'] = df['y'].rolling(window).min().fillna(df['y'])

# If player dropped then climbed back up
df['y_diff_from_5ago'] = df['y'] - df['y'].shift(5)
df['y_climb_after_drop'] = ((df['y_diff_from_5ago'] > 1.0) & (df['deltaY'] > 0)).astype(int)

# Percent of time in the last window where player was on the ground
df['onGround'] = df['onGround'].astype(int)
df['onGround_ratio'] = df['onGround'].rolling(window).mean().fillna(0)

# Convert to ints
df['isFall'] = df['isFall'].astype(int)



# Re Balance Dataframe

ratio = 20
random_state = 42

falls = df[df['isFall'] == 1]
non_falls = df[df['isFall'] == 0]

num_to_keep = min(len(non_falls), len(falls) * ratio)
non_falls_sampled = non_falls.sample(n=num_to_keep, random_state=random_state)

df = pd.concat([falls, non_falls_sampled]).sample(frac=1, random_state=random_state)








# Model Inputs
features = [
    'y', 'deltaY', 'velY', 'onGround',
    #'direction_changed',
    #'low_velocity_duration',
    'recent_y_min',
    'y_climb_after_drop',
    'onGround_ratio'
]
X = df[features]

# Model Output
y = df['isFall']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create new random forest model
model = RandomForestClassifier(class_weight='balanced')

# Train new model on data
model.fit(X_train, y_train)

# Create predictions on testing data
predictions = model.predict(X_test)

# Creates report on predictions
print(classification_report(y_test, predictions))

# Save the model for later
joblib.dump(model, 'models/fall_model.joblib')

print(pd.Series(predictions).value_counts())