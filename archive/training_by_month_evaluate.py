import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import GridSearchCV
import pmdarima as pm
import argparse

sparsity_threshold = 0.2
min_weeks = 65
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # 设置中文字体

def is_model_ok(model_summary):
    print(model_summary)
    summary_text = model_summary.as_text()
    if 'nan' in summary_text or '-inf' in summary_text:
        print("存在 nan 或 -inf")
        return False
    return True

# 添加命令行参数
parser = argparse.ArgumentParser(description='SARIMAX model with specific start date for training')
parser.add_argument('--start_date', type=str, required=True, help='The start date from which to include data for training (format: YYYY-MM-DD)')
args = parser.parse_args()
start_date_filter = pd.to_datetime(args.start_date) 

df = pd.read_csv('final_combined_df.csv')  # 读取数据

# 确保日期格式正确
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.set_index('start_date')
df = df.sort_values('start_date')

# 根据指定的start_date进行过滤
df = df[df.index >= start_date_filter]
print(f"训练数据从 {start_date_filter} 开始")

unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
model_results = []
all_results = pd.DataFrame()

for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    print(f"当前处理的药品名称: {drug_name}, 厂家: {factory_name}")
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    if len(group_data) < min_weeks:
        print(f"跳过 {drug_name} + {factory_name}，数据不足 {min_weeks} 周")
        continue

    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < sparsity_threshold:
        print(f"跳过 {drug_name} + {factory_name}，销量数据过于稀疏（非零值比例: {non_zero_ratio:.2f}）")
        continue

    y = group_data['减少数量']
    log_y = np.log1p(y)

    other_factories = df[(df['药品名称'] == drug_name) & (df['厂家'] != factory_name)]
    for other_factory in other_factories['厂家'].unique():
        other_factory_sales = other_factories[other_factories['厂家'] == other_factory].copy()
        other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)
        other_factory_sales[f'avg_6_period_sales_{other_factory}'] = other_factory_sales['减少数量'].rolling(window=6, min_periods=1).mean().shift(1)
        group_data = pd.merge(group_data, other_factory_sales[[f'previous_sales_{other_factory}', f'avg_6_period_sales_{other_factory}']],
                              left_index=True, right_index=True, how='left')
    group_data.fillna(0, inplace=True)

    for other_factory in other_factories['厂家'].unique():
        group_data[f'previous_sales_{other_factory}'] = np.log1p(group_data[f'previous_sales_{other_factory}'])
        group_data[f'avg_6_period_sales_{other_factory}'] = np.log1p(group_data[f'avg_6_period_sales_{other_factory}'])
    group_data['previous_增加数量'] = np.log1p(group_data['previous_增加数量'])

    exog_cols = ['期初金额占比', 'previous_增加数量'] + [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_period_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()]
    exog = group_data[exog_cols]

    train_end = min_weeks
    history_y = log_y[:train_end]
    history_exog = exog[:train_end]
    
    auto_model = pm.auto_arima(history_y, exogenous=history_exog, seasonal=True, m=52, max_d=2, max_p=3, max_q=3, D=1, stepwise=True, trace=True)
    summary = auto_model.summary()

    if not is_model_ok(summary):
        print(f"自动选取的模型存在异常: {drug_name} + {factory_name}")
        continue

    log_y = log_y.reset_index(drop=True)
    exog = exog.reset_index(drop=True)

    log_predictions = []
    residuals = []

    for t in range(train_end):
        exog_current = exog.iloc[t: t + 1].fillna(0)
        next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
        actual_value = log_y.iloc[t]
        residual = actual_value - next_pred_log
        residuals.append(residual)

    shifted_log_y = log_y.shift(1).fillna(0)
    history_exog_with_y = history_exog.copy()
    history_exog_with_y['shifted_log_y'] = shifted_log_y[:train_end]
    ml_model = RandomForestRegressor()
    ml_model.fit(history_exog_with_y, residuals)
    
    for t in range(train_end, len(log_y)):
        if t >= len(exog):
            print(f"超出索引范围：{t}, 最大值为 {len(exog) - 1}")
            break
        exog_current = exog.iloc[t: t + 1].fillna(0)
        next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()

        exog_current['shifted_log_y'] = log_y.iloc[t - 1]
        predicted_residual = ml_model.predict(exog_current).item() if t > train_end else 0

        if np.isnan(predicted_residual):
            predicted_residual = 0

        final_pred_log = max(next_pred_log + predicted_residual, 0)
        log_predictions.append(final_pred_log)
        actual_value = log_y.iloc[t]
        residual = actual_value - next_pred_log
        residuals.append(residual)

        history_y = log_y[:t + 1]
        history_exog = exog[:t + 1]
        auto_model.update(history_y, exogenous=history_exog)
        history_exog_with_y = history_exog.copy()
        shifted_log_y = log_y.shift(1).fillna(0)
        history_exog_with_y['shifted_log_y'] = shifted_log_y[:t + 1]
        ml_model.fit(history_exog_with_y, residuals[-len(history_exog):])

    predictions = np.expm1(log_predictions)
    actual_values = np.expm1(log_y[train_end:])
    
    group_data['month'] = group_data.index.to_period('M')
    monthly_actual_values = group_data.groupby('month')['减少数量'].sum()

    # 创建一个包含所有月份的索引来确保对齐
    all_months = monthly_actual_values.index

    # 使用相同的月份索引对预测值进行分组和求和
    monthly_predicted_values = pd.Series(predictions, index=group_data.index[train_end:]).groupby(group_data['month'][train_end:]).sum()

    # 确保两个 Series 对齐
    monthly_actual_values = monthly_actual_values.reindex(all_months, fill_value=0)
    monthly_predicted_values = monthly_predicted_values.reindex(all_months, fill_value=0)

    # 计算月度评估指标
    def smape(y_true, y_pred):
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

    mae_monthly = np.mean(np.abs(monthly_actual_values - monthly_predicted_values))
    rmse_monthly = np.sqrt(np.mean((monthly_actual_values - monthly_predicted_values) ** 2))
    smape_monthly = smape(monthly_actual_values, monthly_predicted_values)
    weights = np.where(monthly_actual_values > 0, 1, 0.1)
    weighted_r2_monthly = r2_score(monthly_actual_values, monthly_predicted_values, sample_weight=weights)

    print(f"月度 RMSE: {rmse_monthly}")
    print(f"月度 MAE: {mae_monthly}")
    print(f"月度 SMAPE: {smape_monthly}")
    print(f"月度 R²: {weighted_r2_monthly}")

    model_results.append({
        '药品名称': drug_name,
        '厂家': factory_name,
        'p': auto_model.order[0],
        'd': auto_model.order[1],
        'q': auto_model.order[2],
        'P': auto_model.seasonal_order[0],
        'D': auto_model.seasonal_order[1],
        'Q': auto_model.seasonal_order[2],
        'm': auto_model.seasonal_order[3],
        '月度RMSE': rmse_monthly,
        '月度MAE': mae_monthly,
        '月度R²': weighted_r2_monthly
    })

    # 保存预测结果
    # 确保 '预测类型' 和 '预测减少数量' 列存在
    if '预测类型' not in df.columns:
        df['预测类型'] = ''  # 初始化为空字符串

    if '预测减少数量' not in df.columns:
        df['预测减少数量'] = np.nan  # 初始化为 NaN

    # 对 group_data 进行预测类型和预测结果的更新
    group_data['month'] = group_data.index.to_period('M')

    # 对预测的月份范围进行更新
    prediction_months = monthly_actual_values.index

    # 仅在预测月份范围内更新 '预测类型'
    group_data.loc[group_data['month'].isin(prediction_months), '预测类型'] = '滚动预测'

    # 将每个月的预测结果复制到该月的每一行
    for month in prediction_months:
        # 选取当前月份的数据行并赋值为该月份的预测值
        group_data.loc[group_data['month'] == month, '预测减少数量'] = monthly_predicted_values.loc[month]

    all_results = pd.concat([all_results, group_data])

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(group_data.index, group_data['减少数量'], label='实际减少数量')
    plt.plot(group_data.index[train_end:], predictions, label='滚动预测', linestyle='--')
    
    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('减少数量', fontproperties=font)
    plt.title(f'SARIMAX 滚动预测 vs 实际值 - {drug_name} + {factory_name}', fontproperties=font)
    plt.legend(prop=font)
    plt.pause(3)
    plt.close()

model_results_df = pd.DataFrame(model_results)
model_results_df.to_csv('model_results.csv', index=False)

all_results.to_csv('all_predicted_results.csv', index=False)
