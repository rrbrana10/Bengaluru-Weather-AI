# 🌦️ Bengaluru Weather Prediction AI

A Deep Learning-based meteorological forecasting system using **Stacked LSTM (Long Short-Term Memory)** networks. This project analyzes 26 years of historical data to provide high-accuracy weather forecasts.

* **Rana Kumar** 


---

## 🚀 Project Overview
Traditional weather models are computationally heavy. Our solution uses a data-driven approach:
- **Dataset:** NASA POWER (2000 - 2026)
- **Algorithm:** Stacked LSTM (100 -> 50 Units)
- **Input Window:** 30-day sliding window
- **UI:** Interactive Streamlit Dashboard

## 🛠️ Technical Stack
- **Languages:** Python
- **Frameworks:** TensorFlow, Keras
- **Libraries:** Pandas, NumPy, Scikit-learn, Plotly
- **Deployment:** Streamlit

## 📁 Project Structure
- `app.py`: The Streamlit web application.
- `train.py`: Neural network training script.
- `weather_model.h5`: The trained LSTM model.
- `scaler.pkl`: Min-Max scaler for data normalization.
- `quick_save.py`: Script to generate Excel reports and metrics.

## ⚙️ How to Run
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rrbrana10/Bengaluru-Weather-AI.git](https://github.com/rrbrana10/Bengaluru-Weather-AI.git)