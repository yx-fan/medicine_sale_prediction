import argparse
import pandas as pd
import numpy as np
from prophet import Prophet
from models import configmonth
from models.data_loader import load_data
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

# 定义处理异常值的函数
# def deal_with_outlier(df):
#     """
#     按药品名称和厂家分组处理减少数量列中的异常值。
#     对于超过组内平均值2.5倍或低于0.4倍的值，将其替换为过去5期数据的平均值。
    
#     参数：
#     - df: pandas DataFrame，包含需要处理的减少数量列
    
#     返回：
#     - 处理后的 DataFrame
#     """
#     def replace_outliers(group):
#         # 计算当前分组的平均值
#         mean_value = group['减少数量'].mean()
#         # 设置异常值的上下限
#         upper_limit = mean_value * 4
#         lower_limit = mean_value * 0.25
        
#         # 使用过去3期的平均值替换超过上下限的值
#         for i in range(len(group)):
#             if group['减少数量'].iloc[i] > upper_limit or group['减少数量'].iloc[i] < lower_limit:
#                 # 获取过去3期的平均值
#                 past_3_mean = group['减少数量'].iloc[max(0, i-3):i].mean()
#                 # 如果过去3期没有数据，则使用当前分组的整体平均值
#                 group['减少数量'].iloc[i] = past_3_mean if not np.isnan(past_3_mean) else mean_value
#         return group

#     # 按药品名称和厂家分组并应用替换函数
#     df = df.groupby(['药品名称', '厂家'], group_keys=False).apply(replace_outliers)
#     return df

# Command-line arguments
parser = argparse.ArgumentParser(description='Rolling Prophet model with a specific training start date')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load and filter data
df = load_data('updated_monthly_final_combined_df.csv', start_date_filter, end_date_filter)
# 替换药品名称中的斜杠为下划线
df['药品名称'] = df['药品名称'].str.replace('/', '_')

# Reset index to make 'start_date' a column if it is currently the index
if df.index.name == 'start_date':
    df = df.reset_index()

print("Data columns after loading:", df.columns)
if 'start_date' not in df.columns:
    print("Error: 'start_date' column not found. Available columns are:", df.columns)
    exit(1)

df = df.sort_values(by=['药品名称', '厂家', 'start_date'])

# 设置每组的起始时间为第一个非零数据的日期
def get_first_nonzero_date(group):
    first_nonzero_date = group.loc[group['减少数量'] > 0, 'start_date'].min()
    group = group[group['start_date'] >= first_nonzero_date]  # 过滤数据，只保留从第一个非零日期开始的数据
    return group

df = df.groupby(['药品名称', '厂家'], group_keys=False).apply(get_first_nonzero_date)

# Prepare to store model results
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'prophet_rolling_model_results.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

# Process each group (药品名称 + 厂家)
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # 检查最小月数要求
    if len(group_data) < configmonth.min_months:
        print(f"Skipping {drug_name} + {factory_name}: Insufficient data")
        continue
    
    # 检查稀疏度要求
    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < configmonth.sparsity_threshold:
        print(f"跳过 {drug_name} + {factory_name}: 数据过于稀疏 (非零比率: {non_zero_ratio:.2f})")
        continue
    
    # 调用处理异常值的函数
    # group_data = deal_with_outlier(group_data)

    # Prepare Prophet data format
    group_data = group_data[['start_date', '减少数量']].rename(columns={'start_date': 'ds', '减少数量': 'y'})
    group_data['ds'] = pd.to_datetime(group_data['ds'])  # Ensure date format

    # Rolling prediction setup
    initial_train_size = 5
    rolling_predictions = []
    rolling_actuals = []

    # Loop through each time step for rolling prediction
    for i in range(initial_train_size, len(group_data)):
        train_data = group_data.iloc[:i]  # Use data up to the current point for training
        actual_value = group_data.iloc[i]['y']  # Actual value for the next time step

        # Initialize and fit Prophet model
        model = Prophet()
        model.fit(train_data)

        # Predict the next period (one-step ahead)
        future = model.make_future_dataframe(periods=1, freq='ME')
        forecast = model.predict(future)
        predicted_value = forecast.iloc[-1]['yhat']
        if predicted_value < 0:
           predicted_value = 0

        # Store the prediction and actual value
        rolling_predictions.append(predicted_value)
        rolling_actuals.append(actual_value)

    # Convert rolling predictions and actuals to arrays for metric calculation
    rolling_predictions = np.array(rolling_predictions)
    rolling_actuals = np.array(rolling_actuals)

    # Check if there are enough predictions to calculate metrics
    if len(rolling_predictions) > 0 and len(rolling_actuals) > 0:
        rmse, mae, smape, r2 = calculate_metrics(rolling_actuals, rolling_predictions)

        # Store _ for the current group
        model_result = {
            '药品名称': drug_name, '厂家': factory_name,
            'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
        }
        pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
        print(f"Results for {drug_name} + {factory_name} have been appended to '{results_file}'")
    else:
        print(f"Skipping metrics calculation for {drug_name} + {factory_name} due to insufficient rolling predictions.")

    # Rename 'y' back to '减少数量' for plotting
    group_data = group_data.rename(columns={'y': '减少数量'})

    # 将 'ds' 列设置为索引
    group_data.set_index('ds', inplace=True)

    # Plot predictions
    plot_predictions(group_data, rolling_predictions, drug_name, factory_name, configmonth.font_path, configmonth.plot_dir)

print("All results have been incrementally saved to 'rolling_model_results.csv'")
