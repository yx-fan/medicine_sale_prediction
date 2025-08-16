import argparse
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models import configmonth
from models.data_loader import load_data
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

def main():
    parser = argparse.ArgumentParser(description='Standard SARIMAX Prediction with Time-based Train-Test Split')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
    args = parser.parse_args()

    start_date_filter = pd.to_datetime(args.start_date)
    end_date_filter = pd.to_datetime(args.end_date)

    # Load and filter data
    df = load_data('final_monthly_combined_df_after_cleaning.csv', start_date_filter, end_date_filter)

    if df.empty:
        raise ValueError("The loaded data is empty. Please check the input file and date range.")

    # Group by unique combinations of 药品名称 and 厂家
    unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
    results_file = 'basic_sarimax_model_results.csv'
    results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
    pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

    for _, row in unique_groups.iterrows():
        drug_name = row['药品名称']
        factory_name = row['厂家']
        group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

        # Ensure data is sorted by date
        group_data = group_data.sort_values(by='start_date')

        y = group_data['减少数量']

        # Train-test split: first 50% for training, last 50% for testing
        split_index = len(group_data) // 2
        train_data = y.iloc[:split_index]
        test_data = y.iloc[split_index:]

        try:
            # Train SARIMAX model
            model = SARIMAX(
                train_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)

            # Forecast
            forecast = results.get_forecast(steps=len(test_data))
            predicted_values = forecast.predicted_mean
            predicted_values = np.maximum(predicted_values, 0)  # Ensure non-negative predictions

            # Calculate metrics
            rmse, mae, smape, r2 = calculate_metrics(test_data.values, predicted_values.values)
            print(f"Metrics for {drug_name} + {factory_name} - RMSE: {rmse}, MAE: {mae}, SMAPE: {smape}, R²: {r2}")

            # Save results
            model_result = {
                '药品名称': drug_name,
                '厂家': factory_name,
                'RMSE': rmse,
                'MAE': mae,
                'SMAPE': smape,
                'R²': r2
            }
            pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)

            # Plot predictions
            plot_predictions(group_data, predicted_values, drug_name, factory_name, configmonth.font_path, "model_plots_basic_sarimax")

        except Exception as e:
            print(f"Model training failed for {drug_name} + {factory_name}: {e}")
            # Use simple mean prediction for these groups
            mean_prediction = [train_data.mean()] * len(test_data)
            rmse, mae, smape, r2 = calculate_metrics(test_data.values, mean_prediction)
            model_result = {
                '药品名称': drug_name,
                '厂家': factory_name,
                'RMSE': rmse,
                'MAE': mae,
                'SMAPE': smape,
                'R²': r2
            }
            pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
            continue

    print("All results have been incrementally saved.")

if __name__ == '__main__':
    main()
