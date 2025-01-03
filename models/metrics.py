# metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mask = (np.abs(predicted) + np.abs(actual)) > 0  # 筛选出有效的分母
    smape = np.mean(2 * np.abs(predicted[mask] - actual[mask]) / (np.abs(predicted[mask]) + np.abs(actual[mask]))) * 100
    print(actual, predicted)
    weighted_r2 = r2_score(actual, predicted)
    return rmse, mae, smape, weighted_r2
