# main.py

import argparse
import os
import pandas as pd
import numpy as np
from models import config
from models.data_loader import load_data
from models.model_selection import train_sarimax_model, fit_cnn_model, calculate_residuals
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

# 定义处理异常值的函数
def deal_with_outlier(df):
    """
    按药品名称和厂家分组处理减少数量列中的异常值。
    对于超过组内平均值2.5倍或低于0.4倍的值，将其替换为过去5期数据的平均值。
    
    参数：
    - df: pandas DataFrame，包含需要处理的减少数量列
    
    返回：
    - 处理后的 DataFrame
    """
    def replace_outliers(group):
        # 计算当前分组的平均值
        mean_value = group['减少数量'].mean()
        # 设置异常值的上下限
        upper_limit = mean_value * 2.5
        lower_limit = mean_value * 0.4
        
        # 使用过去3期的平均值替换超过上下限的值
        for i in range(len(group)):
            if group['减少数量'].iloc[i] > upper_limit or group['减少数量'].iloc[i] < lower_limit:
                # 获取过去3期的平均值
                past_3_mean = group['减少数量'].iloc[max(0, i-3):i].mean()
                # 如果过去3期没有数据，则使用当前分组的整体平均值
                group['减少数量'].iloc[i] = past_3_mean if not np.isnan(past_3_mean) else mean_value
        return group

    # 按药品名称和厂家分组并应用替换函数
    df = df.groupby(['药品名称', '厂家'], group_keys=False).apply(replace_outliers)
    return df

# 命令行参数
parser = argparse.ArgumentParser(description='SARIMAX 模型，带有特定的训练开始日期')
parser.add_argument('--start_date', type=str, required=True, help='训练的开始日期 (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='训练的结束日期 (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# 加载并过滤数据
df = load_data('final_combined_df.csv', start_date_filter, end_date_filter)
# 替换药品名称中的斜杠为下划线
df['药品名称'] = df['药品名称'].str.replace('/', '_')
# 准备存储模型结果
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'model_results.csv'

# 创建一个空的 CSV 文件以增量存储结果
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

# 处理每个组（药品名称 + 厂家）
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # 检查最小周数要求
    if len(group_data) < config.min_weeks:
        print(f"跳过 {drug_name} + {factory_name}: 数据不足")
        continue

    # 检查稀疏度要求
    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < config.sparsity_threshold:
        print(f"跳过 {drug_name} + {factory_name}: 数据过于稀疏 (非零比率: {non_zero_ratio:.2f})")
        continue

    # 调用处理异常值的函数
    group_data = deal_with_outlier(group_data)

    # 定义目标变量和外生特征
    y = group_data['减少数量']
    log_y = np.log1p(y)

    # 提取并合并外生变量（其他厂家的前期销售量，6期平均销售量）
    other_factories = df[(df['药品名称'] == drug_name) & (df['厂家'] != factory_name)]
    for other_factory in other_factories['厂家'].unique():
        other_factory_sales = other_factories[other_factories['厂家'] == other_factory].copy()
        other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)
        other_factory_sales[f'avg_6_period_sales_{other_factory}'] = (
            other_factory_sales['减少数量']
            .rolling(window=6, min_periods=1)
            .mean()
            .shift(1)
        )
        group_data = pd.merge(
            group_data,
            other_factory_sales[[f'previous_sales_{other_factory}', f'avg_6_period_sales_{other_factory}']],
            left_index=True, right_index=True, how='left'
        )
    group_data.fillna(0, inplace=True)

    # 外生变量列列表
    exog_cols = ['期初金额占比', 'previous_增加数量'] + \
                [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_period_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()]

    # 过滤 `group_data` 保留外生变量列
    exog = group_data[exog_cols]

    # 准备训练数据
    train_end = config.min_weeks
    history_y = log_y[:train_end]
    history_exog = exog[:train_end]

    # 训练 SARIMAX 模型
    auto_model = train_sarimax_model(history_y, history_exog)

    # 计算残差
    # residuals = calculate_residuals(log_y, exog, auto_model, train_end)
    

    # 使用 CNN 训练残差
    # history_exog = history_exog.astype(float)
    # residuals = np.array(residuals, dtype=float)
    # cnn_model = fit_cnn_model(history_exog, history_y[:train_end], residuals)

    # 使用 SARIMAX + CNN 残差校正进行预测
    predictions = []
    for t in range(train_end, len(log_y)):
        # 获取 SARIMAX 预测值
        exog_current = exog.iloc[t:t + 1].fillna(0)

        # 将最新的 SARIMAX 预测值添加到 exog_current
        next_preds_log = auto_model.predict(n_periods=2, exogenous=exog_current)
        # next_residual = residual_model.predict(n_periods=2, exogenous=exog_current)

        # 在 exog_current 中添加实际观测值（而不是 SARIMAX 的预测值）作为 CNN 的输入
        # actual_y_at_t = log_y.iloc[t - 1]  # 使用上一个时刻的实际值
        # exog_current = np.append(exog_current.values, actual_y_at_t).reshape((1, -1, 1)).astype(np.float32)
        # 计算未来4期的平均值作为当前的预测值
        next_pred_log = np.mean(next_preds_log)
        # next_residual = np.mean(next_residual)
        # 预测残差
        # residual_pred = cnn_model.predict(exog_current, verbose=0).flatten()[0]

        # 结合 SARIMAX 预测和残差预测
        final_pred_log = next_pred_log # + next_residual

        # 加入平滑拟合的上（2倍）下限（0.5倍），使得不要距离平均值过远 20241113
        # 计算拟合值的上下限
        mean_value = log_y[:t].mean()
        upper_limit = mean_value * 2
        lower_limit = mean_value * 0.5
        # 检查预测值是否在合理范围内
        if next_pred_log > upper_limit or next_pred_log < lower_limit:
            # 替换为过去3期数据的平均值
            next_pred_log = log_y[max(0, t-3):t].mean()


        predictions.append(final_pred_log)

        # 用实际观察值更新 SARIMAX
        actual_y_at_t = log_y.iloc[t]
        auto_model.update([actual_y_at_t], exogenous=exog.iloc[t:t+1])

    # 将预测值转换回原始尺度
    predictions = np.expm1(predictions)
    actual_values = np.expm1(log_y[train_end:])

    # 计算评估指标
    rmse, mae, smape, r2 = calculate_metrics(actual_values, predictions)

    # 将当前组的结果存储
    model_result = {
        '药品名称': drug_name, '厂家': factory_name,
        'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
    }

    # 每个组处理后将结果追加到 CSV 文件中
    pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
    print(f"{drug_name} + {factory_name} 的结果已追加到 '{results_file}'")

    # 绘制预测图
    plot_predictions(group_data, predictions, drug_name, factory_name, config.font_path, config.plot_dir)

print("所有结果已逐步保存到 'model_results.csv'")