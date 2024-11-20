import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import configmonth
from models.data_loader import load_data
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

# Command-line arguments
parser = argparse.ArgumentParser(description='Rolling XGBoost model with a specific training start date')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# 加载并筛选数据
df = load_data('updated_monthly_final_combined_df.csv', start_date_filter, end_date_filter)
df['药品名称'] = df['药品名称'].str.replace('/', '_')


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
df.to_excel("output.xlsx", index=False)

# 准备保存模型结果
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'xgv2_model_results.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

# 创建预测结果文件的表头
prediction_file = 'xgv2_prediction_results.csv'
predictions_columns = ['药品名称', '厂家', 'start_date', 'actual', 'prediction']
pd.DataFrame(columns=predictions_columns).to_csv(prediction_file, index=False)

# Process each group
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # 检查最小月数要求
    if len(group_data) < configmonth.min_months:
        print(f"Skipping {drug_name} + {factory_name}: Insufficient data")
        continue

    # 调用处理异常值的函数
    # group_data = deal_with_outlier(group_data)

    # 提取时间特征和滞后特征
    group_data['ds'] = pd.to_datetime(group_data['start_date'])
    group_data['month'] = group_data['ds'].dt.month
    group_data['quarter'] = group_data['ds'].dt.quarter
    group_data['lag_1'] = group_data['减少数量'].shift(1)
    group_data['lag_2'] = group_data['减少数量'].shift(2)
    group_data['lag_3'] = group_data['减少数量'].shift(3)
    group_data['rolling_mean_3'] = group_data['减少数量'].rolling(window=3).mean()
    group_data = group_data.dropna()  # 去除含有缺失值的行

    # 定义特征和目标
    X = group_data[['month', 'quarter', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']]
    y = group_data['减少数量']

    # Rolling prediction setup
    initial_train_size = 15
    rolling_predictions = []
    rolling_actuals = []

    for i in range(initial_train_size, len(group_data)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[[i]]
        actual_value = y.iloc[i]

        # 初始化和训练 XGBoost 模型
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 预测下一个时间步
        predicted_value = model.predict(X_test)[0]
        if predicted_value < 0:
            predicted_value = 0

        # Store the prediction and actual value
        rolling_predictions.append(predicted_value)
        rolling_actuals.append(actual_value)

    # Convert rolling predictions and actuals to arrays for metric calculation
    rolling_predictions = np.array(rolling_predictions)
    rolling_actuals = np.array(rolling_actuals)

# 将每个组的预测结果与实际值合并，并补充缺少的前期实际数据
    prediction_results = []
    for i in range(len(group_data)):
        if i < initial_train_size:
            # 预测开始前的实际数据，没有预测值
            prediction_results.append({
                '药品名称': drug_name,
                '厂家': factory_name,
                'start_date': group_data['ds'].iloc[i],
                'actual': group_data['减少数量'].iloc[i],
                'prediction': np.nan  # 无预测值
            })
        else:
            # 包含实际值和预测值的数据
            prediction_results.append({
                '药品名称': drug_name,
                '厂家': factory_name,
                'start_date': group_data['ds'].iloc[i],
                'actual': group_data['减少数量'].iloc[i],
                'prediction': rolling_predictions[i - initial_train_size]
            })

    # 追加到文件
    pd.DataFrame(prediction_results).to_csv(prediction_file, mode='a', header=False, index=False)


    # Calculate evaluation metrics
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

    # Plot predictions
    plot_predictions(group_data.set_index('ds'), rolling_predictions, drug_name, factory_name, configmonth.font_path, configmonth.plot_dir)

print("All results have been incrementally saved to 'xgv2_model_results.csv'")
