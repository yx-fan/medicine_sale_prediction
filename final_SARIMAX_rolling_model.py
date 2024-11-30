import argparse
import os
import pandas as pd
import numpy as np
from models import config
from models.data_loader import load_data
from models.model_selection import train_sarimax_model_month
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

# Handle outliers
def deal_with_outlier(df):
    """
    Process outliers in the '减少数量' column by grouping by '药品名称'和'厂家'.
    Replace values exceeding 3x or below 0.3333x of the group mean with the mean.
    """
    def replace_outliers(group):
        mean_value = group['减少数量'].mean()
        upper_limit = mean_value * 3
        lower_limit = mean_value * 0.3333
        group['减少数量'] = np.where(
            (group['减少数量'] > upper_limit) | (group['减少数量'] < lower_limit),
            mean_value,
            group['减少数量']
        )
        return group

    df = df.groupby(['药品名称', '厂家'], group_keys=False).apply(replace_outliers)
    return df

# Ensure each group starts from the first non-zero data point
def get_first_nonzero_date(group):
    first_nonzero_date = group.loc[group['减少数量'] > 0, 'start_date'].min()
    group = group[group['start_date'] >= first_nonzero_date]
    return group

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
results_file = 'sarimax_model_results.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

# Process each group (药品名称 + 厂家)
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    group_data = group_data.reset_index().rename(columns={'index': 'start_date'})
    # Ensure each group starts from the first non-zero date
    group_data = get_first_nonzero_date(group_data)
    group_data = group_data.set_index('start_date')

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

    # No external exogenous variables; only internal lagged data (if needed)
    exog = None

    # Prepare training data
    train_end = config.min_months
    history_y = log_y[:train_end]

    # Forecast with Rolling SARIMAX
    predictions = []
    window_size = config.min_months  # 滚动窗口大小

    for t in range(train_end, len(log_y)):
        # 滚动窗口：更新历史数据（移除最早一期，增加最新一期）
        history_y = log_y[t - window_size:t]

        # 更新模型：使用滚动窗口中的数据重新拟合模型
        auto_model = train_sarimax_model_month(history_y, exog)

        # 预测下一期
        next_preds_log = auto_model.predict(n_periods=1)
        predictions.append(next_preds_log[0])

    # 转换预测值回原始尺度
    predictions = np.expm1(predictions)
    actual_values = np.expm1(log_y[train_end:])

    # 计算评估指标
    rmse, mae, smape, r2 = calculate_metrics(actual_values, predictions)

    # Store results for the current group
    model_result = {
        '药品名称': drug_name, '厂家': factory_name,
        'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
    }
    pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
    print(f"Results for {drug_name} + {factory_name} have been appended to '{results_file}'")

    # Plot predictions
    plot_predictions(group_data, predictions, drug_name, factory_name, config.font_path, config.plot_dir_sarimax)

print("All results have been incrementally saved to 'model_results.csv'")
