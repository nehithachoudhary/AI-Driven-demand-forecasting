import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    nonzero = actual != 0
    return {"MAE": float(mean_absolute_error(actual, predicted)), "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))), "MAPE": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else 0.0}