# student-performance-prediction
Student Performance Prediction using Random Forest - Machine Learning Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Загрузка данных из Excel
data = pd.read_excel('StudentPerformance.xls')

# 2. Преобразование категориальных признаков в числовые
data = pd.get_dummies(data, drop_first=True)

# 3. Разделение на признаки и целевую переменную
X = data.drop('G3', axis=1)
y = data['G3']

# 4. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 5. Создание и обучение модели Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Предсказание и оценка модели
predictions = rf_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))

# 7. Важность признаков
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nВажность признаков:")
print(importances.sort_values(ascending=False))
