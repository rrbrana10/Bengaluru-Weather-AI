import streamlit as st
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Weather AI Forecast", layout="wide")
st.title("🌦️ Bengaluru Future Weather Forecaster")

@st.cache_resource
def load_assets():
    return load_model('weather_model.h5', compile=False), joblib.load('scaler.pkl')

model, scaler = load_assets()

# Load and prepare the data
df = pd.read_csv('Bengaluru_2000-2026_weather.csv')
df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j').dt.date

# Get the last available date in your dataset
last_data_date = df['Date'].max()

st.sidebar.info(f"Historical data ends on: {last_data_date}")
user_date = st.sidebar.date_input("Select Date to Predict", value=last_data_date + datetime.timedelta(days=1))

if st.sidebar.button("Predict Forecast"):
    # Determine how many days we need to predict into the future
    days_to_forecast = (user_date - last_data_date).days
    
    if days_to_forecast <= 0:
        # Standard lookup for dates already in the CSV
        idx = df[df['Date'] == user_date].index[0]
        input_data = df.iloc[idx-30:idx].drop(['YEAR', 'DOY', 'Date'], axis=1)
        current_input = scaler.transform(input_data)
        prediction_scaled = model.predict(np.array([current_input]), verbose=0)
    else:
        # RECURSIVE FORECAST for truly upcoming days
        st.warning(f"Generating recursive forecast for {days_to_forecast} days...")
        
        # Start with the last 30 days of real data
        last_30_days = df.iloc[-30:].drop(['YEAR', 'DOY', 'Date'], axis=1)
        current_input = scaler.transform(last_30_days)
        
        # Loop and predict each day one by one
        for _ in range(days_to_forecast):
            # Reshape for LSTM: (1, 30, num_features)
            pred_scaled = model.predict(current_input.reshape(1, 30, -1), verbose=0)
            
            # Slide the window: Remove the oldest day, add the new prediction
            # This uses the model's own output as input for the next day
            new_row = pred_scaled # shape (1, num_features)
            current_input = np.append(current_input[1:], new_row, axis=0)
            
        prediction_scaled = pred_scaled

    # Final Result
    prediction = scaler.inverse_transform(prediction_scaled)[0]

    st.success(f"AI Prediction for {user_date}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Temp", f"{prediction[0]:.2f} °C")
    c2.metric("Humidity", f"{prediction[5]:.2f} %")
    c3.metric("Wind Speed", f"{prediction[3]:.2f} m/s")
    
    st.info("Note: Forecasts further into the future (Recursive) carry higher uncertainty.")