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
    'changepoint_prior_scale': [0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.1, 1.0, 10.0],
}

# 定义处理异常值的函数
# def deal_with_outlier(df):
#     # your implementation...

# Command-line arguments
parser = argparse.ArgumentParser(description='Rolling Prophet model with a specific training start date')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load and filter data
df = load_data('updated_monthly_final_combined_df.csv', start_date_filter, end_date_filter)
df['药品名称'] = df['药品名称'].str.replace('/', '_')

if df.index.name == 'start_date':
    df = df.reset_index()

print("Data columns after loading:", df.columns)
if 'start_date' not in df.columns:
    print("Error: 'start_date' column not found. Available columns are:", df.columns)
    exit(1)

df = df.sort_values(by=['药品名称', '厂家', 'start_date'])

def get_first_nonzero_date(group):
    first_nonzero_date = group.loc[group['减少数量'] > 0, 'start_date'].min()
    group = group[group['start_date'] >= first_nonzero_date]
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

    if len(group_data) < configmonth.min_months:
        print(f"Skipping {drug_name} + {factory_name}: Insufficient data")
        continue

    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < configmonth.sparsity_threshold:
        print(f"Skipping {drug_name} + {factory_name}: 数据过于稀疏 (非零比率: {non_zero_ratio:.2f})")
        continue

    group_data = group_data[['start_date', '减少数量']].rename(columns={'start_date': 'ds', '减少数量': 'y'})
    group_data['ds'] = pd.to_datetime(group_data['ds'])

    # **参数搜索**
    best_params = None
    best_rmse = float('inf')
    train_data = group_data.iloc[:-1]
    actual_value = group_data.iloc[-1]['y']

    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    for params in all_params:
        try:
            model = Prophet(**params)
            model.fit(train_data)

            future = model.make_future_dataframe(periods=1, freq='ME')
            forecast = model.predict(future)
            predicted = forecast.iloc[-1]['yhat']

            if predicted < 0:
                predicted = 0

            rmse = np.sqrt((predicted - actual_value) ** 2)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
        except Exception as e:
            print(f"Error with params {params}: {e}")

    print(f"Best params for {drug_name} + {factory_name}: {best_params}, RMSE: {best_rmse}")

    # **滚动预测**
    rolling_predictions = []
    rolling_actuals = []
    initial_train_size = 5

    for i in range(initial_train_size, len(group_data)):
        train_data = group_data.iloc[:i]
        actual_value = group_data.iloc[i]['y']

        model = Prophet(**best_params)
        model.fit(train_data)

        future = model.make_future_dataframe(periods=1, freq='ME')
        forecast = model.predict(future)
        predicted_value = forecast.iloc[-1]['yhat']
        if predicted_value < 0:
            predicted_value = 0

        rolling_predictions.append(predicted_value)
        rolling_actuals.append(actual_value)

    rolling_predictions = np.array(rolling_predictions)
    rolling_actuals = np.array(rolling_actuals)

    if len(rolling_predictions) > 0 and len(rolling_actuals) > 0:
        rmse, mae, smape, r2 = calculate_metrics(rolling_actuals, rolling_predictions)

        model_result = {
            '药品名称': drug_name, '厂家': factory_name,
            'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
        }
        pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
        print(f"Results for {drug_name} + {factory_name} have been appended to '{results_file}'")
    else:
        print(f"Skipping metrics calculation for {drug_name} + {factory_name} due to insufficient rolling predictions.")

    group_data = group_data.rename(columns={'y': '减少数量'})
    group_data.set_index('ds', inplace=True)
    plot_predictions(group_data, rolling_predictions, drug_name, factory_name, configmonth.font_path, configmonth.plot_dir)

print("All results have been incrementally saved to 'rolling_model_results.csv'")
