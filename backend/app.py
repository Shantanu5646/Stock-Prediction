from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import ta
import traceback
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/lstm_general_stock_model.h5"
SCALERS_DIR = "models/scalers_per_stock"

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("⚠️ Model not found. Make sure lstm_general_stock_model.h5 is in the models folder.")
    model = None

FEATURES = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD',
            'BB_high', 'BB_low', 'ATR', 'OBV', 'Stoch_RSI']
SEQUENCE_LENGTH = 60

@app.route('/')
def home():
    return "Flask backend is running successfully 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker_symbol = data.get("ticker", "").upper()
        if not ticker_symbol:
            return jsonify({"error": "Ticker symbol is required"}), 400

        print(f"🔍 Fetching data for: {ticker_symbol}")

        # Download stock data
        df = yf.download(ticker_symbol, start='2015-01-01', progress=False)
        if df.empty:
            return jsonify({"error": "Invalid ticker or no data found"}), 404

        # Ensure all columns are 1D Series
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        # Technical indicators
        df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        df['EMA_20'] = ta.trend.ema_indicator(close, window=20)
        df['RSI'] = ta.momentum.rsi(close, window=14)
        df['MACD'] = ta.trend.macd_diff(close)
        bb = ta.volatility.BollingerBands(close)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)
        df['OBV'] = ta.volume.on_balance_volume(close, volume)
        df['Stoch_RSI'] = ta.momentum.stochrsi(close, window=14)

        df.dropna(inplace=True)
        if df.shape[0] <= SEQUENCE_LENGTH:
            return jsonify({"error": "Not enough data after preprocessing"}), 400

        # Load scaler
        scaler_path = os.path.join(SCALERS_DIR, f"scaler_{ticker_symbol}.pkl")
        if not os.path.exists(scaler_path):
            return jsonify({"error": f"Scaler for {ticker_symbol} not found"}), 404
        scaler = joblib.load(scaler_path)

        # Scale features
        df_scaled = df.copy()
        df_scaled[FEATURES] = scaler.transform(df[FEATURES])

        # Build sequences
        X = np.array([df_scaled[FEATURES].iloc[i-SEQUENCE_LENGTH:i].values
                      for i in range(SEQUENCE_LENGTH, len(df_scaled))])

        # Predict
        y_pred_scaled = model.predict(X)
        y_pred_scaled = y_pred_scaled.squeeze()  # ensure 1D

        # -----------------------------
        # Inverse transform actual and predicted Close prices
        # -----------------------------
        # Prepare full arrays for inverse transform
        y_full = np.zeros((len(df_scaled), len(FEATURES)))
        y_full[:, FEATURES.index('Close')] = df_scaled['Close'].values.flatten()  # scaled actual

        pred_full = np.zeros((len(y_pred_scaled), len(FEATURES)))
        pred_full[:, FEATURES.index('Close')] = y_pred_scaled.flatten()  # predicted scaled

        # Inverse transform
        y_actual_orig = scaler.inverse_transform(y_full[SEQUENCE_LENGTH:])[:, FEATURES.index('Close')]
        y_pred_orig = scaler.inverse_transform(pred_full)[:, FEATURES.index('Close')]

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_actual_orig, y_pred_orig))
        mae = mean_absolute_error(y_actual_orig, y_pred_orig)
        mape = np.mean(np.abs((y_actual_orig - y_pred_orig) / np.maximum(y_actual_orig, 1e-8))) * 100
        accuracy = 100 - mape

        # Dates
        chart_dates_full = df.index[SEQUENCE_LENGTH:].strftime('%Y-%m-%d').tolist()
        chart_dates_short = chart_dates_full[-30:]
        actual_short = y_actual_orig[-30:].tolist()
        predicted_short = y_pred_orig[-30:].tolist()

        # Next-day prediction
        last_seq = df_scaled[FEATURES].iloc[-SEQUENCE_LENGTH:].values[np.newaxis, ...]
        next_scaled = model.predict(last_seq).squeeze()
        next_full = np.zeros((1, len(FEATURES)))
        next_full[0, FEATURES.index('Close')] = next_scaled
        next_day_price = scaler.inverse_transform(next_full)[0, FEATURES.index('Close')]

        # Current stock info
        stock_info = yf.Ticker(ticker_symbol)
        current_price = stock_info.history(period="1d")['Close'][-1]
        company_name = stock_info.info.get('longName', ticker_symbol)
        last_training_time = "2025-11-15 14:00"
        model_version = "v1.0"

        return jsonify({
            "ticker": ticker_symbol,
            "company": company_name,
            "current_price": float(current_price),
            "next_day_price": float(next_day_price),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "accuracy": float(accuracy),
            "chart_dates_short": chart_dates_short,
            "actual_short": actual_short,
            "predicted_short": predicted_short,
            "chart_dates_full": chart_dates_full,
            "actual_full": y_actual_orig.tolist(),
            "predicted_full": y_pred_orig.tolist(),
            "last_training_time": last_training_time,
            "model_version": model_version
               
        })

    except Exception as e:
        print("❌ Error:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Error occurred while predicting. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)
