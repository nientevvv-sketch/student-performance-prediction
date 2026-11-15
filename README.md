# Student Performance Prediction

A machine learning project that predicts students' final grades using a Random Forest regression model. The script reads student data from an Excel workbook, converts categorical columns into numeric indicators, trains and evaluates the model, and prints feature importances.

## Requirements

- Python 3
- An Excel dataset named `StudentPerformance.xls` in the project directory, with a final-grade column named `G3`

Install the required packages:

```bash
python -m pip install pandas scikit-learn xlrd
```

`xlrd` is needed by pandas to read legacy `.xls` files.

## Run

Run the prediction script from the directory containing the dataset:

```bash
python student-performance-prediction.py
```

The script reports mean absolute error (MAE), the R² score, and feature importances for the test data.

## How it works

1. Loads `StudentPerformance.xls` with pandas.
2. Converts categorical features to numeric columns using one-hot encoding.
3. Uses `G3` as the target and the remaining columns as input features.
4. Splits the data into training and test sets (70%/30%).
5. Trains a Random Forest regressor and evaluates its predictions.

## Model code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset from Excel.
data = pd.read_excel("StudentPerformance.xls")

# Convert categorical features to numeric columns.
data = pd.get_dummies(data, drop_first=True)

# Separate input features from the target grade.
X = data.drop("G3", axis=1)
y = data["G3"]

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train the Random Forest model.
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=5, random_state=42
)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model.
predictions = rf_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))

# Display feature importances, from highest to lowest.
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature importance:")
print(importances.sort_values(ascending=False))
```
