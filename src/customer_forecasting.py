import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import CUSTOMER_MODELS_DIR
from src.evaluation import metrics

LOOKBACK = 30
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
# With a 30-step lookback and int(0.60 * n) training records, n=54 gives
# two training sequences, plus non-empty chronological validation and test sets.
MIN_CUSTOMER_RECORDS = 54


def model_directory(user_id, product_id):
    return CUSTOMER_MODELS_DIR / f"user_{int(user_id)}" / f"product_{int(product_id)}"


def customer_model_paths(user_id, product_id):
    directory = model_directory(user_id, product_id)
    return directory, directory / "lstm.keras", directory / "scaler.joblib", directory / "metadata.json"


def _sequences(values, split_index):
    scaled = values
    train_x, train_y, validation_x, validation_y, test_x, test_y = [], [], [], [], [], []
    validation_end = split_index[1]
    for index in range(LOOKBACK, len(scaled)):
        window = scaled[index - LOOKBACK:index]
        target = scaled[index]
        if index < split_index[0]:
            train_x.append(window); train_y.append(target)
        elif index < validation_end:
            validation_x.append(window); validation_y.append(target)
        else:
            test_x.append(window); test_y.append(target)
    return tuple(np.asarray(part) for part in (train_x, train_y, validation_x, validation_y, test_x, test_y))


def train_customer_lstm(sales, user_id, product_id):
    import tensorflow as tf

    series = sales.sort_values("sale_date").copy()
    series["sale_date"] = pd.to_datetime(series["sale_date"], errors="raise")
    record_count = len(series)
    if record_count < MIN_CUSTOMER_RECORDS:
        raise ValueError(f"Insufficient historical data for customer-specific LSTM training. Current records: {record_count}. Required records: {MIN_CUSTOMER_RECORDS}.")

    values = series["quantity"].to_numpy(dtype="float32").reshape(-1, 1)
    train_end = int(record_count * TRAIN_FRACTION)
    validation_end = int(record_count * (TRAIN_FRACTION + VALIDATION_FRACTION))
    scaler = MinMaxScaler().fit(values[:train_end])
    scaled = scaler.transform(values)
    train_x, train_y, validation_x, validation_y, test_x, test_y = _sequences(scaled, (train_end, validation_end))
    if min(len(train_x), len(validation_x), len(test_x)) < 1:
        raise ValueError("Insufficient historical data to create chronological train, validation, and test sequences.")

    model = tf.keras.Sequential([tf.keras.layers.Input((LOOKBACK, 1)), tf.keras.layers.LSTM(24), tf.keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=30, batch_size=8, shuffle=False, callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)], verbose=0)
    test_predictions = scaler.inverse_transform(model.predict(test_x, verbose=0)).ravel()
    test_actual = scaler.inverse_transform(test_y).ravel()
    score = metrics(test_actual, test_predictions)
    directory, model_path, scaler_path, metadata_path = customer_model_paths(user_id, product_id)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    metadata = {"user_id": int(user_id), "product_id": int(product_id), "model_type": "LSTM", "training_record_count": record_count, "sequence_length": LOOKBACK, "train_records": train_end, "validation_records": validation_end - train_end, "test_records": record_count - validation_end, "training_date": pd.Timestamp.utcnow().isoformat(), "last_data_date_used": series["sale_date"].max().date().isoformat(), "metrics": score}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_customer_model(user_id, product_id):
    directory, model_path, scaler_path, metadata_path = customer_model_paths(user_id, product_id)
    if not all(path.exists() for path in (model_path, scaler_path, metadata_path)):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(model_path), joblib.load(scaler_path), json.loads(metadata_path.read_text(encoding="utf-8"))


def customer_model_status(sales, user_id, product_id):
    model_data = load_customer_model(user_id, product_id)
    record_count = len(sales)
    latest_date = str(pd.to_datetime(sales["sale_date"]).max().date()) if record_count else None
    if model_data is None:
        return {"status": "not_trained", "record_count": record_count, "required_records": MIN_CUSTOMER_RECORDS, "last_trained": None, "training_record_count": None, "stale": False}
    _, _, metadata = model_data
    stale = record_count > metadata["training_record_count"] or (latest_date and latest_date > metadata["last_data_date_used"])
    return {"status": "stale" if stale else "trained", "record_count": record_count, "required_records": MIN_CUSTOMER_RECORDS, "last_trained": metadata["training_date"], "training_record_count": metadata["training_record_count"], "last_data_date_used": metadata["last_data_date_used"], "metrics": metadata["metrics"], "stale": stale}


def forecast_customer_lstm(sales, user_id, product_id, horizon):
    model_data = load_customer_model(user_id, product_id)
    if model_data is None:
        raise ValueError("Customer-specific model has not been trained yet.")
    model, scaler, metadata = model_data
    history = sales.sort_values("sale_date")
    if len(history) < LOOKBACK:
        raise ValueError(f"Insufficient historical data: at least {LOOKBACK} observations are required.")
    values = history["quantity"].to_numpy(dtype="float32").reshape(-1, 1).tolist()
    last_date = pd.to_datetime(history["sale_date"]).max()
    result = []
    for offset in range(1, horizon + 1):
        window = np.asarray(values[-LOOKBACK:], dtype="float32").reshape(-1, 1)
        scaled_window = scaler.transform(window).reshape(1, LOOKBACK, 1)
        prediction = max(0.0, float(scaler.inverse_transform(model.predict(scaled_window, verbose=0))[0, 0]))
        values.append([prediction])
        result.append({"Date": (last_date + pd.Timedelta(days=offset)).date().isoformat(), "Predicted Demand": prediction})
    return result