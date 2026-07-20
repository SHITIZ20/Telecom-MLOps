# Generated from: MLOps_Lab1.ipynb
# Converted at: 2026-07-13T05:17:21.103Z

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

# FIX: Changed from absolute Windows path to a clean, relative path
df = pd.read_csv("Telecom_Tower_Failure_Dataset_10000-1 (1).csv")

# Clean trailing spaces from columns just in case
df.columns = df.columns.str.strip()

# Safely drop target columns
columns_to_drop = [col for col in ['Tower_ID', 'Failure_Within_48Hrs'] if col in df.columns]
X = df.drop(columns=columns_to_drop)
y = df['Failure_Within_48Hrs']

# FIX: Automatically convert any text columns to numbers so sklearn doesn't crash
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Accuracy: ", acc)

joblib.dump(model, "telecom_tower_model.pkl")
metrics = {"accuracy": acc}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Training Completed Successfully")
