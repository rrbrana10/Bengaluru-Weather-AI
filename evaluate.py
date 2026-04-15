import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 1. Load everything
model = load_model('weather_model.h5', compile=False)
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('Bengaluru_2000-2026_weather.csv')

# 2. Prep data (numeric only)
data = df.drop(['YEAR', 'DOY'], axis=1)
scaled_data = scaler.transform(data)

# 3. Create test sequences (using the last 20% of data)
window_size = 30
X, y = [], []
for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size])
X, y = np.array(X), np.array(y)

# Use only the last 500 days for a quick test
X_test = X[-500:]
y_test_actual_scaled = y[-500:]

# 4. Make Predictions
predictions_scaled = model.predict(X_test)

# 5. Inverse Scale to get real values
y_test_actual = scaler.inverse_transform(y_test_actual_scaled)
y_test_pred = scaler.inverse_transform(predictions_scaled)

# 6. Calculate Metrics (Focusing on Temperature - Index 0)
mae = mean_absolute_error(y_test_actual[:, 0], y_test_pred[:, 0])
rmse = np.sqrt(mean_squared_error(y_test_actual[:, 0], y_test_pred[:, 0]))

print("\n" + "="*30)
print(f"ACCURACY REPORT (Temperature)")
print(f"Mean Absolute Error (MAE): {mae:.4f} °C")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} °C")
print("="*30)

# 7. Visual Comparison Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:100, 0], label='Actual Temperature', color='blue', alpha=0.7)
plt.plot(y_test_pred[:100, 0], label='Predicted Temperature', color='red', linestyle='--')
plt.title('Accuracy Check: Actual vs. Predicted (Next 100 Days)')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()