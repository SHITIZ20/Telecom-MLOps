import joblib 
import pandas as pd 
model=joblib.load("telecom_tower_model.pkl") 
new_data=pd.DataFrame({ 
"Battery_Voltage":[51], 
"Temperature_C":[43], 
"Signal_Strength_Percent":[71], 
"Power_Consumption_W":[1800], 
"Fan_Speed_RPM":[2800], 
"Humidity_Percent":[60],
"Traffic_Load":[2500],
"Tower_Age_Years":[8] 
}) 
prediction=model.predict(new_data) 
if prediction[0]==1: 
   print("Hardware Failure Predicted") 
else: 
   print("Tower is Healthy")