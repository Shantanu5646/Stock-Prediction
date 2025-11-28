import React, { useState, useRef } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import html2canvas from "html2canvas";
import "./App.css";

function App() {
  const [ticker, setTicker] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const shortChartRef = useRef(null);
  const fullChartRef = useRef(null);

  // ----------------------------
  // Fetch prediction from backend
  // ----------------------------
  const handlePredict = async () => {
    if (!ticker) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", { ticker });
      setResult(response.data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert("Error occurred while predicting. Make sure ticker is valid and backend is running.");
    }

    setLoading(false);
  };

  // ----------------------------
  // Prepare Last 30 Days Chart
  // ----------------------------
  const shortChartData = [];
  if (result?.chart_dates_short) {
    for (let i = 0; i < result.chart_dates_short.length; i++) {
      shortChartData.push({
        date: result.chart_dates_short[i],
        Actual: result.actual_short[i],
        Predicted: result.predicted_short[i],
      });
    }
  }

  // ----------------------------
  // Prepare Full History Chart
  // ----------------------------
  const fullChartData = [];
  if (result?.chart_dates_full) {
    for (let i = 0; i < result.chart_dates_full.length; i++) {
      fullChartData.push({
        date: result.chart_dates_full[i],
        Actual: result.actual_full[i],
        Predicted: result.predicted_full[i],
      });
    }
  }

  // ----------------------------
  // Download chart as PNG
  // ----------------------------
  const handleDownloadChart = async (ref, filename) => {
    if (!ref.current) {
      alert("Graph not ready");
      return;
    }
    try {
      const canvas = await html2canvas(ref.current);
      const link = document.createElement("a");
      link.download = filename;
      link.href = canvas.toDataURL("image/png");
      link.click();
    } catch (err) {
      console.error(err);
      alert("Failed to download chart");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "30px" }}>
      <h1>📈 Stock Price Predictor</h1>

      {/* Input + Predict */}
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value.toUpperCase())}
        placeholder="Enter Stock Symbol (e.g. TSLA)"
        style={{ padding: "8px", marginRight: "10px" }}
      />
      <button onClick={handlePredict} style={{ padding: "8px 15px" }}>
        Predict
      </button>

      {/* Loading Spinner */}
      {loading && (
        <div style={{ marginTop: "30px" }}>
          <div className="loading-container">
            <div className="spinner">
              </div>
              </div>
          <div className="loading-text">
            <h3 style={{ marginTop: "20px" }}>Predicting… Please wait</h3></div>
        </div>
      )}

      {/* Model Metrics */}
      {result && !loading && (
        <div
          style={{
            marginTop: "20px",
            textAlign: "left",
            maxWidth: "600px",
            margin: "20px auto",
            border: "1px solid #ddd",
            padding: "20px",
            borderRadius: "8px",
          }}
        >
          <h3>📊 Model Insights</h3>
          <p>RMSE: {result.rmse.toFixed(2)}</p>
          <p>MAE: {result.mae.toFixed(2)}</p>
          <p>MAPE: {result.mape.toFixed(2)}%</p>
          <p>Accuracy: {result.accuracy.toFixed(2)}%</p>
          <p>Last Training: {result.last_training_time}</p>
          <p>Model Version: {result.model_version}</p>
        </div>
      )}

      {/* Stock Info */}
      {result && !loading && (
        <>
          <h2>
            {result.company} ({result.ticker})
          </h2>
          <p>Current Price: ${result.current_price.toFixed(2)}</p>
          <p>Next Day Predicted Price: ${result.next_day_price.toFixed(2)}</p>
        </>
      )}

      {/* ----------------------------
          Last 30 Days Graph
      ---------------------------- */}
      {!loading && shortChartData.length > 0 && (
        <div
          ref={shortChartRef}
          style={{
            marginTop: "30px",
            padding: "20px",
            border: "1px solid #ddd",
            borderRadius: "10px",
          }}
        >
          <h2>📉 Last 30 Days — Actual vs Predicted</h2>
          <ResponsiveContainer width="95%" height={350}>
            <LineChart data={shortChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(str) => {
                  const date = new Date(str);
                  return `${date.getMonth() + 1}/${date.getDate()}`;
                }}
                interval={4}
              />
              <YAxis />
              <Tooltip
                labelFormatter={(label) => {
                  const d = new Date(label);
                  return `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="Actual" stroke="blue" dot={false} />
              <Line type="monotone" dataKey="Predicted" stroke="red" dot={false} />
            </LineChart>
          </ResponsiveContainer>

          <button
            onClick={() => handleDownloadChart(shortChartRef, "last_30_days_chart.png")}
            className="download-btn"
          >
            Download Last 30 Days Chart
          </button>
        </div>
      )}

      {/* ----------------------------
          Full History Graph
      ---------------------------- */}
      {!loading && fullChartData.length > 0 && (
        <div
          ref={fullChartRef}
          style={{
            marginTop: "50px",
            padding: "20px",
            border: "1px solid #ddd",
            borderRadius: "10px",
          }}
        >
          <h2>📊 Full History — Actual vs Predicted (2015 → Today)</h2>
          <ResponsiveContainer width="95%" height={450}>
            <LineChart data={fullChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(str) => {
                  const date = new Date(str);
                  return `${date.getMonth() + 1}/${date.getDate()}`;
                }}
                interval={60}
              />
              <YAxis />
              <Tooltip
                labelFormatter={(label) => {
                  const d = new Date(label);
                  return `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="Actual" stroke="blue" dot={false} />
              <Line type="monotone" dataKey="Predicted" stroke="red" dot={false} />
            </LineChart>
          </ResponsiveContainer>

          <button
            onClick={() => handleDownloadChart(fullChartRef, "full_history_chart.png")}
            className="download-btn"
          >
            Download Full History Chart
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
