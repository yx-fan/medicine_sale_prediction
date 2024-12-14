import argparse
import os
import pandas as pd
import numpy as np
from models import config
from models.data_loader import load_data
from models.model_selection import train_sarimax_model_month
from models.metrics import calculate_metrics
from models.visualization import plot_predictions
from scipy import stats

def enhance_feature_engineering(df):
    df = df.sort_index()

    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df['减少数量'].shift(lag)
    
    window_sizes = [3, 6, 12]
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df['减少数量'].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df['减少数量'].rolling(window=window, min_periods=1).std()
    
    df['ewma_3'] = df['减少数量'].ewm(span=3, adjust=False).mean()
    df['pct_change_1'] = df['减少数量'].pct_change()
    df['pct_change_3'] = df['减少数量'].pct_change(periods=3)
    df['trend_strength'] = df['减少数量'].diff().abs().rolling(window=6).mean()
    df['volatility'] = df['减少数量'].rolling(window=6).std()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['month_sin'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['month_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Advanced SARIMAX Prediction with Enhanced Features')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
    args = parser.parse_args()

    start_date_filter = pd.to_datetime(args.start_date)
    end_date_filter = pd.to_datetime(args.end_date)

    df = load_data('final_monthly_combined_df_after_cleaning.csv', start_date_filter, end_date_filter)

    if df.empty:
        raise ValueError("The loaded data is empty. Please check the input file and date range.")
    
    unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
    results_file = 'advanced_sarimax_model_results.csv'
    results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
    pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

    for _, row in unique_groups.iterrows():
        drug_name = row['药品名称']
        factory_name = row['厂家']
        group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

        group_data = enhance_feature_engineering(group_data)

        y = group_data['减少数量']
        log_y = np.log1p(y)

        exog_cols = [
            'lag_1', 'lag_3', 'lag_6', 'lag_12',
            'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
            'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
            'ewma_3', 'pct_change_1', 'pct_change_3',
            'trend_strength', 'volatility',
            'month_sin', 'month_cos', 'is_outlier'
        ]
        exog = group_data[exog_cols]

        train_end = config.min_months
        history_y = log_y[:train_end]
        history_exog = exog[:train_end]

        predictions = []
        adaptive_window = train_end

        for t in range(train_end, len(log_y)):
            adaptive_window = max(config.min_months, min(len(log_y), adaptive_window))
            recent_y = log_y.iloc[t - adaptive_window:t]
            recent_exog = exog.iloc[t - adaptive_window:t]

            if len(recent_y) < config.min_months or recent_exog.empty:
                print(f"Skipping step at index {t}: insufficient data for training.")
                continue

            try:
                auto_model = train_sarimax_model_month(recent_y, recent_exog)
                next_preds_log = auto_model.predict(n_periods=1, exogenous=exog.iloc[t:t+1])
                predictions.append(next_preds_log.iloc[0])
            except Exception as e:
                print(f"Model training failed at step {t} due to: {e}")
                predictions.append(0)

        if len(predictions) == 0 or len(log_y[train_end:]) == 0:
            print(f"Skipping metrics calculation for {drug_name} + {factory_name}: insufficient data.")
            continue

        predictions = np.maximum(np.expm1(predictions), 0)
        actual_values = np.expm1(log_y[train_end:])
        rmse, mae, smape, r2 = calculate_metrics(actual_values, predictions)

        model_result = {
            '药品名称': drug_name, '厂家': factory_name,
            'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
        }
        pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
        print(f"Results for {drug_name} + {factory_name} appended to '{results_file}'")

        plot_predictions(group_data, predictions, drug_name, factory_name, config.font_path, config.plot_dir_sarimax)

    print("All results saved to 'v2_advanced_sarimax_model_results.csv'")

if __name__ == '__main__':
    main()
