import argparse
import os
import pandas as pd
import numpy as np
from models import config
from models.data_loader import load_data
from models.model_selection import train_sarimax_model, fit_cnn_model, calculate_residuals, train_sarimax_model_month
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

# Define the function to handle outliers
def deal_with_outlier(df):
    """
    Process outliers in the '减少数量' column by grouping by '药品名称' and '厂家'.
    Replace values exceeding 2x or below 0.5x of the group mean with the mean.
    """
    def replace_outliers(group):
        mean_value = group['减少数量'].mean()
        upper_limit = mean_value * 2
        lower_limit = mean_value * 0.5
        group['减少数量'] = np.where(
            (group['减少数量'] > upper_limit) | (group['减少数量'] < lower_limit),
            mean_value,
            group['减少数量']
        )
        return group

    df = df.groupby(['药品名称', '厂家'], group_keys=False).apply(replace_outliers)
    return df

# Command-line arguments
parser = argparse.ArgumentParser(description='SARIMAX model with a specific training start date')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load and filter data
df = load_data('updated_monthly_final_combined_df.csv', start_date_filter, end_date_filter)

# Prepare to store model results
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'model_results.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

# Process each group (药品名称 + 厂家)
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # Check minimum month requirement
    if len(group_data) < config.min_months:
        print(f"Skipping {drug_name} + {factory_name}: Insufficient data")
        continue

    # Check sparsity requirement
    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < config.sparsity_threshold:
        print(f"Skipping {drug_name} + {factory_name}: Data too sparse (non-zero ratio: {non_zero_ratio:.2f})")
        continue

    # Process outliers
    group_data = deal_with_outlier(group_data)

    # Define target variable and exogenous features
    y = group_data['减少数量']
    log_y = np.log1p(y)

    # Extract and merge exogenous variables (previous month’s sales for other manufacturers, 6-month average)
    other_factories = df[(df['药品名称'] == drug_name) & (df['厂家'] != factory_name)]
    for other_factory in other_factories['厂家'].unique():
        other_factory_sales = other_factories[other_factories['厂家'] == other_factory].copy()
        other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)
        other_factory_sales[f'avg_6_month_sales_{other_factory}'] = (
            other_factory_sales['减少数量']
            .rolling(window=6, min_periods=1)
            .mean()
            .shift(1)
        )
        group_data = pd.merge(
            group_data,
            other_factory_sales[[f'previous_sales_{other_factory}', f'avg_6_month_sales_{other_factory}']],
            left_index=True, right_index=True, how='left'
        )
    group_data.fillna(0, inplace=True)

    # Define exogenous variables
    exog_cols = ['previous_增加数量'] + \
                [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_month_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()]

    exog = group_data[exog_cols]

    # Prepare training data
    train_end = config.min_months
    history_y = log_y[:train_end]
    history_exog = exog[:train_end]

    # Train SARIMAX model
    auto_model = train_sarimax_model_month(history_y, history_exog)

    # Forecast with SARIMAX + CNN residual correction
    predictions = []
    for t in range(train_end, len(log_y)):
        exog_current = exog.iloc[t:t + 1].fillna(0)

        # SARIMAX forecast
        next_preds_log = auto_model.predict(n_periods=2, exogenous=exog_current)
        next_pred_log = np.mean(next_preds_log)

        final_pred_log = next_pred_log  # Combine predictions here as needed
        predictions.append(final_pred_log)

        # Update SARIMAX with actual value
        actual_y_at_t = log_y.iloc[t]
        auto_model.update([actual_y_at_t], exogenous=exog.iloc[t:t + 1])

    # Convert predictions back to the original scale
    predictions = np.expm1(predictions)
    actual_values = np.expm1(log_y[train_end:])

    # Calculate evaluation metrics
    rmse, mae, smape, r2 = calculate_metrics(actual_values, predictions)

    # Store results for the current group
    model_result = {
        '药品名称': drug_name, '厂家': factory_name,
        'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
    }
    pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
    print(f"Results for {drug_name} + {factory_name} have been appended to '{results_file}'")

    # Plot predictions
    plot_predictions(group_data, predictions, drug_name, factory_name, config.font_path, config.plot_dir)

print("All results have been incrementally saved to 'model_results.csv'")
