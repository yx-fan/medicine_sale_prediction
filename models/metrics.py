# metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual))) * 100
    print(actual, predicted)
    weighted_r2 = r2_score(actual, predicted)
    return rmse, mae, smape, weighted_r2
