from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

from src.data_factory import ensure_time_series_dataset


WINDOW_SIZE = 14


def build_supervised_windows(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(window_size, len(values)):
        X.append(values[idx - window_size : idx])
        y.append(values[idx])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def run_pipeline(base_dir: str | Path) -> dict:
    base_path = Path(base_dir)
    dataset_path = ensure_time_series_dataset(base_path)
    dataframe = pd.read_csv(dataset_path)

    values = dataframe["demand"].to_numpy(dtype="float32")
    X, y = build_supervised_windows(values, WINDOW_SIZE)

    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    runtime_mode = "tensorflow_keras"
    log_dir = base_path / "logs" / "fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    history_artifact = log_dir / "history.json"

    artifacts_dir = base_path / "artifacts"
    processed_dir = base_path / "data" / "processed"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    model_artifact = artifacts_dir / "best_model.joblib"
    forecast_artifact = processed_dir / "forecast_values.csv"
    report_artifact = processed_dir / "time_series_forecasting_report.json"

    try:
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        X_train_tf = X_train.reshape((-1, WINDOW_SIZE, 1))
        X_test_tf = X_test.reshape((-1, WINDOW_SIZE, 1))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
        )
        history = model.fit(
            X_train_tf,
            y_train,
            validation_split=0.2,
            epochs=30,
            batch_size=16,
            verbose=0,
            callbacks=[tensorboard_callback],
        )
        predictions = model.predict(X_test_tf, verbose=0).reshape(-1)
        history_payload = {
            "loss": [round(float(v), 4) for v in history.history.get("loss", [])],
            "val_loss": [round(float(v), 4) for v in history.history.get("val_loss", [])],
            "mae": [round(float(v), 4) for v in history.history.get("mae", [])],
            "val_mae": [round(float(v), 4) for v in history.history.get("val_mae", [])],
        }
        history_artifact.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        model.save(artifacts_dir / "keras_forecasting_model.keras")
        model_artifact = artifacts_dir / "keras_forecasting_model.keras"
    except Exception:
        runtime_mode = "fallback_without_tensorflow"
        fallback_model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42, max_iter=800)
        fallback_model.fit(X_train, y_train)
        predictions = fallback_model.predict(X_test)
        history_payload = {
            "runtime_mode": runtime_mode,
            "note": "TensorFlow was not available in the local environment; fallback regressor was used.",
        }
        history_artifact.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(fallback_model, model_artifact)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mape = float(np.mean(np.abs((y_test - predictions) / y_test)) * 100)

    forecast_df = pd.DataFrame(
        {
            "actual": np.round(y_test, 4),
            "predicted": np.round(predictions, 4),
        }
    )
    forecast_df.to_csv(forecast_artifact, index=False)

    summary = {
        "runtime_mode": runtime_mode,
        "row_count": int(len(dataframe)),
        "window_size": WINDOW_SIZE,
        "train_window_count": int(len(X_train)),
        "test_window_count": int(len(X_test)),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
        "dataset_artifact": str(dataset_path),
        "forecast_artifact": str(forecast_artifact),
        "model_artifact": str(model_artifact),
        "tensorboard_log_dir": str(log_dir),
        "history_artifact": str(history_artifact),
        "report_artifact": str(report_artifact),
    }
    report_artifact.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
