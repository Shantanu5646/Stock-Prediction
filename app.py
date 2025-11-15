from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import ta
import traceback
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)
CORS(app)  # enable CORS for frontend

# Load your trained model
MODEL_PATH = "lstm_stock_model.h5"

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("⚠️ Model not found. Make sure lstm_stock_model.h5 is in the same folder.")
    model = None

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
        stock_data = yf.download(ticker_symbol, start='2015-01-01')

        if stock_data.empty:
            return jsonify({"error": "Invalid ticker or no data found"}), 404

        # Prepare data like in notebook
        close_series = pd.Series(stock_data['Close'].values.flatten(), index=stock_data.index)
        high_series = pd.Series(stock_data['High'].values.flatten(), index=stock_data.index)
        low_series = pd.Series(stock_data['Low'].values.flatten(), index=stock_data.index)
        volume_series = pd.Series(stock_data['Volume'].values.flatten(), index=stock_data.index)

        # Add technical indicators
        stock_data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
        stock_data['EMA_20'] = ta.trend.ema_indicator(close_series, window=20)
        stock_data['RSI'] = ta.momentum.rsi(close_series, window=14)
        stock_data['MACD'] = ta.trend.macd_diff(close_series)
        bb = ta.volatility.BollingerBands(close_series)
        stock_data['BB_high'] = bb.bollinger_hband()
        stock_data['BB_low'] = bb.bollinger_lband()
        stock_data['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series, window=14)
        stock_data['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
        stock_data['Stoch_RSI'] = ta.momentum.stochrsi(close_series, window=14)

        # Drop NaN rows
        stock_data.dropna(inplace=True)

        # Feature scaling
        features = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_high', 'BB_low', 'ATR', 'OBV', 'Stoch_RSI']
        scaler = MinMaxScaler()
        scaled_array = scaler.fit_transform(stock_data[features])
        scaled_df = pd.DataFrame(scaled_array, columns=features, index=stock_data.index)

        # Prepare sequences for model
        sequence_length = 100
        X_test = []
        for i in range(sequence_length, len(scaled_df)):
            X_test.append(scaled_df.iloc[i-sequence_length:i].values)
        X_test = np.array(X_test)

        # Predict on last sequences
        y_pred_scaled = model.predict(X_test)

        # Inverse transform
        dummy_pred = np.zeros((y_pred_scaled.shape[0], len(features)))
        dummy_pred[:, 0] = y_pred_scaled[:, 0]
        y_pred_orig = scaler.inverse_transform(dummy_pred)[:, 0]

        # Actual Close prices
        y_actual_orig = stock_data['Close'].iloc[sequence_length:].values

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_actual_orig, y_pred_orig))
        mae = mean_absolute_error(y_actual_orig, y_pred_orig)
        mape = np.mean(np.abs((y_actual_orig - y_pred_orig) / np.maximum(y_actual_orig, 1e-8))) * 100
        accuracy = 100 - (rmse / np.mean(y_actual_orig)) * 100

        # Dates for chart
        chart_dates_full = stock_data.index[sequence_length:].strftime('%Y-%m-%d').tolist()

        # Last 30 days data for short chart
        chart_dates_short = chart_dates_full[-30:]
        actual_short = y_actual_orig[-30:].tolist()
        predicted_short = y_pred_orig[-30:].tolist()

        # Full history data
        actual_full = y_actual_orig.tolist()
        predicted_full = y_pred_orig.tolist()

        # Last sequence for next-day prediction
        last_sequence = scaled_df[-sequence_length:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        next_scaled = model.predict(last_sequence)
        next_full = np.zeros((1, len(features)))
        next_full[0, 0] = next_scaled
        next_day_price = scaler.inverse_transform(next_full)[0, 0]

        # Current stock price and company
        stock_info = yf.Ticker(ticker_symbol)
        current_price = stock_info.history(period="1d")['Close'][-1]
        company_name = stock_info.info.get('longName', ticker_symbol)

        # Optional metadata
        last_training_time = "2025-10-15 12:00:00"
        model_version = "LSTM_v1.0"

        return jsonify({
            "ticker": ticker_symbol,
            "company": company_name,
            "current_price": float(current_price),
            "next_day_price": float(next_day_price),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "accuracy": float(accuracy),
            "last_training_time": last_training_time,
            "model_version": model_version,
            # Last 30 days
            "chart_dates_short": chart_dates_short,
            "actual_short": actual_short,
            "predicted_short": predicted_short,
            # Full history
            "chart_dates_full": chart_dates_full,
            "actual_full": actual_full,
            "predicted_full": predicted_full
        })

    except Exception as e:
        print("❌ Error:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": "Error occurred while predicting. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)
