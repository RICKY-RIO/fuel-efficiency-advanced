import pandas as pd
import joblib

# Load models and scaler
rf = joblib.load('rf_model.pkl')
svr = joblib.load('svr_model.pkl')
gbr = joblib.load('gbr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load new cars data
new_cars = pd.read_csv('new_cars.csv')

# Features to use (must match training)
features = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
X_new = new_cars[features]
X_new_scaled = scaler.transform(X_new)

# Predict for each model
rf_pred = rf.predict(X_new_scaled)
svr_pred = svr.predict(X_new_scaled)
gbr_pred = gbr.predict(X_new_scaled)

# Add predictions to DataFrame
new_cars['RF_Predicted_MPG'] = rf_pred
new_cars['SVR_Predicted_MPG'] = svr_pred
new_cars['GBR_Predicted_MPG'] = gbr_pred

# Print or save results
print(new_cars[['Car Name', 'RF_Predicted_MPG', 'SVR_Predicted_MPG', 'GBR_Predicted_MPG']])

# Optionally, save to CSV
new_cars.to_csv('new_cars_with_predictions.csv', index=False)
