import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Read in data from csv
df = pd.read_csv('data/fall_data_example.csv')
print(df.head())
print("--------------------------")

# Add a deltaY feature
df['deltaY'] = df['y'].diff().fillna(0)

# Convert to ints
df['isFall'] = df['isFall'].astype(int)

# Model Inputs
X = df[['y', 'deltaY', 'velY', 'onGround']]

# Model Output
y = df['isFall']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create new random forest model
model = RandomForestClassifier()

# Train new model on data
model.fit(X_train, y_train)

# Create predictions on testing data
predictions = model.predict(X_test)

# Creates report on predictions
print(classification_report(y_test, predictions))

# Save the model for later
joblib.dump(model, '../models/fall_model.joblib')