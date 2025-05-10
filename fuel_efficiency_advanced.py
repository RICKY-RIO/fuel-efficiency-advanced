import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



# =========================
# 1. LOAD AND CLEAN DATASET
# =========================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
columns = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
           'Acceleration', 'Model Year', 'Origin', 'Car Name']
df = pd.read_csv(url, names=columns, delim_whitespace=True, na_values='?')
df.dropna(inplace=True)
df['Horsepower'] = df['Horsepower'].astype(float)

# =========================
# 2. ADD A NEW CAR (MARUTI ALTO)
# =========================
new_car_df = pd.DataFrame({
    'MPG': [None],  # Use the real MPG if you know it, else None/np.nan
    'Cylinders': [3],
    'Displacement': [796],
    'Horsepower': [47.3],
    'Weight': [850],
    'Acceleration': [15.5],
    'Model Year': [2025],
    'Origin': [3],
    'Car Name': ['maruti alto']
})
df = pd.concat([df, new_car_df], ignore_index=True)
df.reset_index(drop=True, inplace=True)  # Ensure indices are sequential

# =========================
# 3. FEATURE SELECTION & SCALING
# =========================
features = ['Cylinders', 'Displacement', 'Horsepower', 'Weight',
            'Acceleration', 'Model Year', 'Origin']
X = df[features]
y = df['MPG']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 4. TRAIN-TEST SPLIT
#    (Exclude new car if MPG is unknown)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled[~y.isna()], y[~y.isna()], test_size=0.2, random_state=42
)

# =========================
# 5. TRAIN MODELS
# =========================
# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)



# Support Vector Regression (SVM)
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)

# Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)

# =========================
# 5b. SAVE MODELS AND SCALER
# =========================
import joblib
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(svr, 'svr_model.pkl')
joblib.dump(gbr, 'gbr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models and scaler saved successfully!")

# =========================
# 6. EVALUATE MODELS
# =========================
print("\n========== RANDOM FOREST REGRESSOR ==========")
print("Random Forest R2:", r2_score(y_test, rf_pred))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

print("\n========== SUPPORT VECTOR REGRESSION (SVM) ==========")
print("SVR R2:", r2_score(y_test, svr_pred))
print("SVR RMSE:", np.sqrt(mean_squared_error(y_test, svr_pred)))

print("\n========== GRADIENT BOOSTING REGRESSOR ==========")
print("Gradient Boosting R2:", r2_score(y_test, gbr_pred))
print("Gradient Boosting RMSE:", np.sqrt(mean_squared_error(y_test, gbr_pred)))

# =========================
# 7. VISUALIZE PREDICTIONS
# =========================
plt.figure(figsize=(10, 5))
plt.scatter(y_test, rf_pred, alpha=0.6, label='Random Forest')
plt.scatter(y_test, svr_pred, alpha=0.6, label='SVR')
plt.scatter(y_test, gbr_pred, alpha=0.6, label='Gradient Boosting')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.legend()
plt.title('Actual vs Predicted MPG')
plt.show()

# =========================
# 8. PRINT CAR NAMES WITH PREDICTIONS FOR EACH MODEL
# =========================
car_names_test = df.loc[y_test.index, 'Car Name'].values

print("\n========== RANDOM FOREST REGRESSOR ==========")
print("Car Name                  | Actual MPG | Predicted MPG")
print("-" * 55)
for name, actual, pred in zip(car_names_test, y_test, rf_pred):
    print(f"{name:25s} | {actual:10.2f} | {pred:13.2f}")

print("\n========== SUPPORT VECTOR REGRESSION (SVM) ==========")
print("Car Name                  | Actual MPG | Predicted MPG")
print("-" * 55)
for name, actual, pred in zip(car_names_test, y_test, svr_pred):
    print(f"{name:25s} | {actual:10.2f} | {pred:13.2f}")

print("\n========== GRADIENT BOOSTING REGRESSOR ==========")
print("Car Name                  | Actual MPG | Predicted MPG")
print("-" * 55)
for name, actual, pred in zip(car_names_test, y_test, gbr_pred):
    print(f"{name:25s} | {actual:10.2f} | {pred:13.2f}")

# =========================
# 9. PREDICT MPG FOR A NEW CAR (E.G., MARUTI ALTO)
# =========================
alto_features = [[3, 796, 47.3, 850, 15.5, 2025, 3]]
alto_scaled = scaler.transform(alto_features)
alto_pred_rf = rf.predict(alto_scaled)
alto_pred_svr = svr.predict(alto_scaled)
alto_pred_gbr = gbr.predict(alto_scaled)

print("\n========== PREDICTED MPG FOR MARUTI ALTO ==========")
print(f"Random Forest:            {alto_pred_rf[0]:.2f} MPG")
print(f"Support Vector Regression:{alto_pred_svr[0]:.2f} MPG")
print(f"Gradient Boosting:        {alto_pred_gbr[0]:.2f} MPG")


with open('paste.txt', 'w') as f:
    # Random Forest
    f.write("===== RANDOM FOREST REGRESSOR =====\n")
    f.write("Car Name                  | Actual MPG | Predicted MPG\n")
    f.write("-" * 55 + "\n")
    for name, actual, pred in zip(car_names_test, y_test, rf_pred):
        f.write(f"{name:25s} | {actual:10.2f} | {pred:13.2f}\n")
    f.write("\n")

    # SVM
    f.write("===== SUPPORT VECTOR REGRESSION (SVM) =====\n")
    f.write("Car Name                  | Actual MPG | Predicted MPG\n")
    f.write("-" * 55 + "\n")
    for name, actual, pred in zip(car_names_test, y_test, svr_pred):
        f.write(f"{name:25s} | {actual:10.2f} | {pred:13.2f}\n")
    f.write("\n")

    # Gradient Boosting
    f.write("===== GRADIENT BOOSTING REGRESSOR =====\n")
    f.write("Car Name                  | Actual MPG | Predicted MPG\n")
    f.write("-" * 55 + "\n")
    for name, actual, pred in zip(car_names_test, y_test, gbr_pred):
        f.write(f"{name:25s} | {actual:10.2f} | {pred:13.2f}\n")
