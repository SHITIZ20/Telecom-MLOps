# Generated from: MLOps_Lab1.ipynb
# Converted at: 2026-07-13T05:17:21.103Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

df = pd.read_csv(r"C:\Users\nmims.student\Desktop\Carol\Lab 1\Telecom_Tower_Failure_Dataset_10000-1.csv")

df

X = df.drop(columns=['Tower_ID','Failure_Within_48Hrs'])
y = df['Failure_Within_48Hrs']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model= RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

acc = model.score(X_test,y_test)
print("Accuracy: ", acc)

joblib.dump(model,"telecom_tower_model.pkl")
metrics={"accuracy":acc}
with open("metrics.json","w") as f:
    json.dump(metrics,f,indent=4)
print("Training Completed Successfully")