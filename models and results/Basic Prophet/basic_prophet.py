import argparse
import pandas as pd
import numpy as np
from prophet import Prophet
from models import configmonth
from models.data_loader import load_data
from models.metrics import calculate_metrics
from models.visualization import plot_predictions
from itertools import product

# 定义参数网格
param_grid = {
    'seasonality_mode': ['additive', 'multiplicative'],
    'changepoint_prior_scale': [0.1],
    'seasonality_prior_scale': [1.0],
}

# Command-line arguments
parser = argparse.ArgumentParser(description='Rolling Prophet model with a specific training start date')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load and filter data
df = load_data('final_monthly_combined_df_after_cleaning.csv', start_date_filter, end_date_filter)
df['药品名称'] = df['药品名称'].str.replace('/', '_')

if df.index.name == 'start_date':
    df = df.reset_index()

print("Data columns after loading:", df.columns)
if 'start_date' not in df.columns:
    print("Error: 'start_date' column not found. Available columns are:", df.columns)
    exit(1)

df = df.sort_values(by=['药品名称', '厂家', 'start_date'])

# Prepare to store model results
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'basic_prophet_model_results.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

# Process each group (药品名称 + 厂家)
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    group_data = group_data[['start_date', '减少数量']].rename(columns={'start_date': 'ds', '减少数量': 'y'})
    group_data['ds'] = pd.to_datetime(group_data['ds'])

    # 按时间点划分训练集和测试集
    split_index = int(len(group_data) * 0.5)  # 前 50% 数据作为训练集
    train_data = group_data.iloc[:split_index]
    test_data = group_data.iloc[split_index:]

    if len(train_data) == 0 or len(test_data) == 0:
        print(f"Skipping group {drug_name} + {factory_name}: Not enough data for training and testing.")
        continue

    # 使用最佳参数进行训练
    best_params = {
        'seasonality_mode': 'additive',
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 1.0
    }
    model = Prophet(**best_params)
    model.fit(train_data)

    # 生成预测数据
    future = model.make_future_dataframe(periods=len(test_data), freq='ME')
    forecast = model.predict(future)

    # 提取预测值
    predicted_values = forecast.iloc[-len(test_data):]['yhat'].values
    actual_values = test_data['y'].values

    # 确保预测值为非负
    predicted_values = np.maximum(predicted_values, 0)

    # 计算误差
    rmse, mae, smape, r2 = calculate_metrics(actual_values, predicted_values)
    print(f"Metrics for {drug_name} + {factory_name} - RMSE: {rmse}, MAE: {mae}, SMAPE: {smape}, R²: {r2}")

    # 保存结果
    model_result = {
        '药品名称': drug_name, '厂家': factory_name,
        'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
    }
    pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
    print(f"Results for {drug_name} + {factory_name} have been appended to '{results_file}'")

    # 绘制预测结果
    group_data = group_data.rename(columns={'y': '减少数量'})
    group_data.set_index('ds', inplace=True)
    plot_predictions(group_data, predicted_values, drug_name, factory_name, configmonth.font_path, "model_plots_basic_prophet")

print("All results have been incrementally saved to 'results.csv'")
