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

sparsity_threshold = 0.7     # 设置稀疏数据阈值
min_weeks = 65    # 定义最小周数
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # 设置中文字体

# 定义函数，检查模型是否正常
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

# 读取数据
df = pd.read_csv('final_combined_df.csv') 

# 确保日期格式正确
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.set_index('start_date')
df = df.sort_values('start_date')

# 根据指定的start_date进行过滤
df = df[df.index >= start_date_filter]
print(f"训练数据从 {start_date_filter} 开始")

# 设置新表格，用于保存模型参数和误差信息
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
model_results = []   # 用于保存模型参数和误差信息的表格
all_results = pd.DataFrame()

# 遍历每个 药品名称 + 厂家 组合
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    print(f"当前处理的药品名称: {drug_name}, 厂家: {factory_name}")
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # 如果数据不足 65 周，则跳过
    if len(group_data) < min_weeks:
        print(f"跳过 {drug_name} + {factory_name}，数据不足 {min_weeks} 周")
        continue

    # 如果销量数据过于稀疏，则跳过
    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < sparsity_threshold: # 如果非零值比例小于阈值，则跳过
        print(f"跳过 {drug_name} + {factory_name}，销量数据过于稀疏（非零值比例: {non_zero_ratio:.2f})")
        continue

    # 定义目标变量和外生变量
    y = group_data['减少数量']
    log_y = np.log1p(y)

    # 判断是否连续 4 周减少数量为 0
    recent_data = group_data.iloc[-12 * 4:]  # 获取最近12个月（假设每月有4周）
    consecutive_zeros = (recent_data['减少数量'] == 0).astype(int).groupby(recent_data['减少数量'].ne(0).cumsum()).cumsum()
    use_rolling_forecast = (consecutive_zeros >= 4).any()

    # 获取同一药品下的其他药厂数据，并创建外生变量previous_sales_{other_factory}和avg_6_period_sales_{other_factory}
    other_factories = df[(df['药品名称'] == drug_name) & (df['厂家'] != factory_name)]
    for other_factory in other_factories['厂家'].unique():   # 遍历每个其他药厂，计算其上一期销量和上六期平均销量
        other_factory_sales = other_factories[other_factories['厂家'] == other_factory].copy()
        other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)
        other_factory_sales[f'avg_6_period_sales_{other_factory}'] = other_factory_sales['减少数量'].rolling(window=6, min_periods=1).mean().shift(1)
        group_data = pd.merge(group_data, other_factory_sales[[f'previous_sales_{other_factory}', f'avg_6_period_sales_{other_factory}']],
                              left_index=True, right_index=True, how='left')
    group_data.fillna(0, inplace=True)  # 填充缺失值

    # 提取月份信息
    group_data['month'] = group_data.index.month
    group_data = pd.get_dummies(group_data, columns=['month'], prefix='month') 

    # 将所有 month 列转换为整数类型（0 和 1）
    month_cols = [col for col in group_data.columns if col.startswith('month_')]
    group_data[month_cols] = group_data[month_cols].astype(int)

    # 对除期初金额占比之外的外生变量进行log处理
    for other_factory in other_factories['厂家'].unique():
        group_data[f'previous_sales_{other_factory}'] = np.log1p(group_data[f'previous_sales_{other_factory}'])
        group_data[f'avg_6_period_sales_{other_factory}'] = np.log1p(group_data[f'avg_6_period_sales_{other_factory}'])
    group_data['previous_增加数量'] = np.log1p(group_data['previous_增加数量'])

    # 构建外生变量
    exog_cols = ['期初金额占比', 'previous_增加数量'] + [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_period_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + month_cols
    print("===================== 外生变量 =====================")
    print(exog_cols)
    exog = group_data[exog_cols]

    # 初始化模型，使用前 min_weeks 的数据进行初始训练
    train_end = min_weeks
    history_y = log_y[:train_end]  # 使用 log_y
    history_exog = exog[:train_end]
    print("===================== 历史减少数量 =====================")
    print(history_y.head())
    print("===================== 历史外生变量 =====================")
    print(history_exog.head())

    # 使用 pmdarima 进行自动模型选择
    print("===================== 自动模型选择 =====================")
    auto_model = pm.auto_arima(history_y, exogenous=history_exog, seasonal=True, m=52, max_d=2, max_p=3, max_q=3, D=1, stepwise=True, trace=True)
    summary = auto_model.summary()

    # 检查模型是否正常
    if is_model_ok(summary):
        print(f"自动选取的模型: {drug_name} + {factory_name}")
    else:
        print(f"自动选取的模型存在异常: {drug_name} + {factory_name}")
        continue

    # 重置索引，确保索引正确
    # log_y = log_y.reset_index(drop=True)
    # exog = exog.reset_index(drop=True)
    residuals = []
    # 使用 SARIMAX 对训练集进行完整预测，获取残差
    for t in range(train_end):
        exog_current = exog.iloc[t: t+1].fillna(0)
        next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
        actual_y_at_t = log_y.iloc[t]
        auto_model.update([actual_y_at_t], exogenous=exog_current)  # 更新 SARIMAX 模型
        residual = actual_y_at_t - next_pred_log  # 计算残差
        residuals.append(residual)

    # 进行滚动预测
    if use_rolling_forecast:
        print(f"使用滚动预测: {drug_name} + {factory_name}")
        log_predictions = []
    
        # 使用历史残差和历史外生变量训练机器学习模型
        shifted_log_y = log_y.shift(1).fillna(0)
        history_exog_with_y = history_exog.copy()
        history_exog_with_y['shifted_log_y'] = shifted_log_y[:train_end]
        ml_model = RandomForestRegressor()
        ml_model.fit(history_exog_with_y, residuals)

        for t in range(train_end, len(log_y)):
            if t >= len(exog):
                break
            exog_current = exog.iloc[t: t + 1].fillna(0)
            try:
                next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
            except KeyError as e:
                print(f"索引错误: {e}, 当前 t 值: {t}, exog.iloc[t:t+1] 出错")

            exog_current['shifted_log_y'] = log_y.iloc[t - 1]
            if t > train_end:
                predicted_residual = ml_model.predict(exog_current).item()
            else:
                predicted_residual = 0

            if np.isnan(predicted_residual):
                predicted_residual = 0

            recent_residuals = residuals[-3:]
            if len(recent_residuals) > 0:
                scaling_factor = 1 * np.abs(recent_residuals[-1]) / np.mean(np.abs(recent_residuals))
            else:
                scaling_factor = 1

            residuals_series = pd.Series(residuals)
            if len(residuals) >= 3:
                residuals_series = pd.Series(residuals[-3:])
                smoothed_residual = 1 * predicted_residual + 0 * residuals_series.mean()
            else:
                smoothed_residual = predicted_residual

            final_pred_log = max(next_pred_log + smoothed_residual * scaling_factor, 0)
            final_pred_log = min(final_pred_log, 1.5 * next_pred_log)
            print(f"Final prediction: {final_pred_log}")
            print(f"Next prediction: {next_pred_log}")
            print(f"Predicted residual: {final_pred_log - next_pred_log}")

            if np.isnan(final_pred_log):
                final_pred_log = 0

            log_predictions.append(final_pred_log)
            actual_value = log_y.iloc[t]
            residual = actual_value - next_pred_log
            residuals.append(residual)

            # 滚动窗口：将真实的观测值加入训练集
            history_y = log_y[:t+1]
            history_exog = exog[:t+1]

            auto_model.update(history_y, exogenous=history_exog)
            history_exog_with_y = history_exog.copy()
            shifted_log_y = log_y.shift(1).fillna(0)
            history_exog_with_y['shifted_log_y'] = shifted_log_y[:t + 1]
            ml_model.fit(history_exog_with_y, residuals[-len(history_exog):])


    else:
        # 使用静态预测
        print(f"使用静态预测: {drug_name} + {factory_name}")

        # 初始化保存预测结果的列表
        log_predictions = []

        # 遍历测试集的每个时间点，逐个进行预测
        for t in range(train_end, len(log_y)):
            exog_current = exog.iloc[t: t + 1].fillna(0)  # 取出当前时间点的外生变量

            try:
                # 使用模型预测当前时刻
                next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
                
                # 更新模型状态，类似滚动更新，但不包含新数据到训练集，仅更新模型状态
                actual_y_at_t = log_y.iloc[t]
                auto_model.update([actual_y_at_t], exogenous=exog_current)
            except KeyError as e:
                print(f"索引错误: {e}, 当前 t 值: {t}, exog.iloc[t:t+1] 出错")
                continue

            # 将预测结果保存到列表中
            log_predictions.append(next_pred_log)


    predictions = np.expm1(log_predictions)
    actual_values = np.expm1(log_y[train_end:])
    predicted_values = predictions
    mean_actual = np.mean(actual_values)

    def smape(y_true, y_pred):
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

    mae = np.mean(np.abs(actual_values - predicted_values))
    mae_percentage = (mae / mean_actual) * 100
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    smape_value = smape(actual_values, predicted_values)
    weights = np.where(actual_values > 0, 1, 0.1)
    weighted_r2 = r2_score(actual_values, predicted_values, sample_weight=weights)
    r2_log = r2_score(np.log1p(actual_values), np.log1p(predicted_values))

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAE%: {mae_percentage}")
    print(f"SMAPE: {smape_value}")
    print(f"R²: {weighted_r2}")
    print(f"R² (log): {r2_log}")

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
        'AIC': auto_model.aic(),
        'BIC': auto_model.bic(),
        '测试集RMSE': rmse,
        '测试集MAE': mae,
        '测试集R²': weighted_r2
    })

    if '预测类型' not in group_data.columns:
        group_data['预测类型'] = ''
    if '预测减少数量' not in group_data.columns:
        group_data['预测减少数量'] = np.nan

    group_data.iloc[train_end:, group_data.columns.get_loc('预测类型')] = '滚动预测' if use_rolling_forecast else '静态预测'
    group_data.iloc[train_end:, group_data.columns.get_loc('预测减少数量')] = pd.Series(predictions).values

    all_results = pd.concat([all_results, group_data])

    plt.figure(figsize=(10, 6))
    plt.plot(group_data.index, group_data['减少数量'], label='实际减少数量')
    plt.plot(group_data.index[train_end:], predictions, label='滚动预测' if use_rolling_forecast else '静态预测', linestyle='--')

    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('减少数量', fontproperties=font)
    plt.title(f'SARIMAX 滚动预测 vs 实际值 - {drug_name} + {factory_name}', fontproperties=font)
    plt.legend(prop=font)
    plt.pause(3)
    plt.close()

model_results_df = pd.DataFrame(model_results)
model_results_df.to_csv('model_results.csv', index=False)
all_results.to_csv('all_predicted_results.csv', index=False)