import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import joblib

# Load and Process Data
df = pd.read_csv('Bengaluru_2000-2026_weather.csv')

def get_date(row):
    return datetime.datetime(int(row['YEAR']), 1, 1) + datetime.timedelta(int(row['DOY']) - 1)

df['Date'] = df.apply(get_date, axis=1)
df.set_index('Date', inplace=True)
data = df.drop(['YEAR', 'DOY'], axis=1)

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, 'scaler.pkl')

# Windowing (30 days)
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(y.shape[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
print("Training Model...")
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
model.save('weather_model.h5')
print("Saved: weather_model.h5 and scaler.pkl")


# Save the numerical results of every epoch cycle
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_metrics.csv', index=False)
print("Success: Epoch metrics saved to training_metrics.csv")