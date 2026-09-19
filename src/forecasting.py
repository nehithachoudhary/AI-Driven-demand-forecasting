from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.evaluation import metrics
from src.feature_engineering import FEATURE_COLUMNS, add_features


def train_tree_models(featured, train_end, models_dir):
    train = featured[featured["Date"] <= train_end]
    test = featured[featured["Date"] > train_end]
    x_train, y_train = train[FEATURE_COLUMNS], train["Demand"]
    x_test, y_test = test[FEATURE_COLUMNS], test["Demand"]
    rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(x_train, y_train)
    xgb = XGBRegressor(n_estimators=250, max_depth=7, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", n_jobs=4, random_state=42)
    xgb.fit(x_train, y_train)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, models_dir / "random_forest.joblib")
    joblib.dump(xgb, models_dir / "xgboost.joblib")
    return {"Random Forest": (test, rf.predict(x_test)), "XGBoost": (test, xgb.predict(x_test))}


def train_sarima(series, train_end, models_dir):
    train = series[series["Date"] <= train_end].set_index("Date")["Demand"].asfreq("D")
    test = series[series["Date"] > train_end]
    fitted = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    prediction = fitted.forecast(steps=len(test))
    models_dir.mkdir(parents=True, exist_ok=True)
    fitted.save(models_dir / "sarima_reference.pkl")
    return test, np.asarray(prediction), {"store_id": str(series["Store ID"].iloc[0]), "product_id": str(series["Product ID"].iloc[0])}


def train_lstm(series, train_end, models_dir, lookback=30):
    import tensorflow as tf
    values = series.sort_values("Date")["Demand"].to_numpy(dtype="float32").reshape(-1, 1)
    split = int((series["Date"] <= train_end).sum())
    scaler = MinMaxScaler().fit(values[:split])
    scaled = scaler.transform(values)
    x_train, y_train, x_test, y_test = [], [], [], []
    for index in range(lookback, len(scaled)):
        window = scaled[index - lookback:index]
        if index < split:
            x_train.append(window); y_train.append(scaled[index])
        else:
            x_test.append(window); y_test.append(scaled[index])
    validation_size = max(1, int(len(x_train) * 0.1))
    x_fit, y_fit = np.asarray(x_train[:-validation_size]), np.asarray(y_train[:-validation_size])
    x_validation, y_validation = np.asarray(x_train[-validation_size:]), np.asarray(y_train[-validation_size:])
    model = tf.keras.Sequential([tf.keras.layers.Input((lookback, 1)), tf.keras.layers.LSTM(24), tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_fit, y_fit, epochs=8, batch_size=32, validation_data=(x_validation, y_validation), shuffle=False, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=0)
    predictions = scaler.inverse_transform(model.predict(np.asarray(x_test), verbose=0)).ravel()
    actual = series[series["Date"] > train_end].copy()
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "lstm.keras")
    joblib.dump({"scaler": scaler, "lookback": lookback, "store_id": str(series["Store ID"].iloc[0]), "product_id": str(series["Product ID"].iloc[0])}, models_dir / "lstm_scaler.joblib")
    return actual, predictions


def recursive_lstm_forecast(df, store_id, product_id, horizon, model, scaler, lookback=30):
    history = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].sort_values("Date")
    if len(history) < lookback:
        raise ValueError(f"Insufficient history: at least {lookback} observations are required")
    values = history["Demand"].to_numpy(dtype="float32").reshape(-1, 1).tolist()
    predictions = []
    for _ in range(horizon):
        window = np.asarray(values[-lookback:], dtype="float32").reshape(-1, 1)
        scaled_window = scaler.transform(window).reshape(1, lookback, 1)
        prediction = max(0.0, float(scaler.inverse_transform(model.predict(scaled_window, verbose=0))[0, 0]))
        date = history["Date"].iloc[-1] + pd.Timedelta(days=len(predictions) + 1)
        values.append([prediction])
        predictions.append({"Date": date.strftime("%Y-%m-%d"), "Predicted Demand": prediction})
    return predictions


def recursive_tree_forecast(df, store_id, product_id, horizon, model):
    history = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)].sort_values("Date").copy()
    if len(history) < 31:
        raise ValueError("Insufficient history: at least 31 observations are required")
    predictions = []
    last_units = float(history["Units Sold"].iloc[-1])
    for _ in range(horizon):
        date = history["Date"].iloc[-1] + pd.Timedelta(days=1)
        row = {"Date": date, "Store ID": store_id, "Product ID": product_id, "Units Sold": last_units, "Demand": np.nan}
        expanded = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        engineered = add_features(expanded, drop_na=False).iloc[[-1]]
        prediction = max(0.0, float(model.predict(engineered[FEATURE_COLUMNS])[0]))
        row["Demand"] = prediction
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        predictions.append({"Date": date.strftime("%Y-%m-%d"), "Predicted Demand": prediction})
    return predictions


def choose_best(results):
    comparison = []
    for model_name, (actual_frame, prediction) in results.items():
        score = metrics(actual_frame["Demand"], prediction)
        comparison.append({"Model": model_name, **score})
    return pd.DataFrame(comparison).sort_values("MAE").reset_index(drop=True)