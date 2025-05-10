# Fuel Efficiency Prediction Project 🚗

This project predicts the fuel efficiency (MPG) of cars using machine learning models in Python. You can train the models, add new cars for prediction, and view results in your browser-all with simple scripts!

---

## Features

- **Train and evaluate models** (Random Forest, SVR, Gradient Boosting)
- **Predict MPG for new cars** by adding them to a CSV file
- **View results in your browser** as a stylish table
- **Reusable saved models** for fast predictions

---

## How to Use

1. **Train the Models**

python fuel_efficiency_advanced.py

text
- Trains models and saves them as `.pkl` files.

2. **Add New Cars**
- Open `new_cars.csv` and add your new car details (one per row).

3. **Predict MPG for New Cars**

python predict_new_cars.py

text
- Generates `new_cars_with_predictions.csv` with predicted MPG.

4. **View Predictions in Your Browser**

python show_new_car_predictions.py

text
- Opens a table of predictions in your browser.

---

## Files in This Project

- `fuel_efficiency_advanced.py` - Train and evaluate models
- `predict_new_cars.py` - Predict MPG for new cars
- `new_cars.csv` - Add your new cars here
- `new_cars_with_predictions.csv` - Predicted MPG for new cars
- `show_new_car_predictions.py` - View predictions in browser
- `.pkl` files - Saved machine learning models
- `paste.txt` - Test set results (not used for new cars)
- `README.md` - This file!

---

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib

Install with:
pip install pandas scikit-learn joblib

text

---

## Example

| Car Name         | RF_Predicted_MPG | SVR_Predicted_MPG | GBR_Predicted_MPG |
|------------------|------------------|-------------------|-------------------|
| hyundai i20      | 31.62            | 22.48             | 40.55             |
| maruti swift     | 33.20            | 22.48             | 39.99             |
| ford mustang eco | 22.77            | 22.48             | 33.02             |

---

## License

MIT (or add your license here)

---

**Made with ❤️ by RIKHITH**
