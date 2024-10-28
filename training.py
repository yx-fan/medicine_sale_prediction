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
    # 将 Summary 对象转换为字符串
    print(model_summary)
    summary_text = model_summary.as_text()

    # 解析 summary 文本，找到异常情况
    if 'nan' in summary_text or '-inf' in summary_text:
        print("存在 nan 或 -inf")
        return False
    return True

# 定义超参数网格
#param_grid = {
#    'n_estimators': [100, 200, 300],
#    'max_depth': [5, 10, 20, None],
#    'min_samples_split': [2, 10, 20],
#    'min_samples_leaf': [1, 5, 10],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'bootstrap': [True, False]
#}

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
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')  # 获取唯一的 药品名称 + 厂家 组合
model_results = []   # 用于保存模型参数和误差信息的表格
all_results = pd.DataFrame()   # 遍历每个 药品名称 + 厂家 组合

# 遍历每个 药品名称 + 厂家 组合
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    print(f"当前处理的药品名称: {drug_name}, 厂家: {factory_name}")
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()  # 取出该组合的所有数据

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
    df['month'] = df.index.month
    df = pd.get_dummies(df, columns=['month'], prefix='month') 

    # 对除期初金额占比之外的外生变量进行log处理
    for other_factory in other_factories['厂家'].unique():
        group_data[f'previous_sales_{other_factory}'] = np.log1p(group_data[f'previous_sales_{other_factory}'])
        group_data[f'avg_6_period_sales_{other_factory}'] = np.log1p(group_data[f'avg_6_period_sales_{other_factory}'])
    group_data['previous_增加数量'] = np.log1p(group_data['previous_增加数量'])

    # 构建外生变量
    exog_cols = ['期初金额占比', 'previous_增加数量'] + [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_period_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + [f'month_{i}' for i in range(1, 13)]
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
    auto_model = pm.auto_arima(history_y, exogenous=history_exog, seasonal=True, m=52, max_d=2, max_p=3, max_q=3, D=1, stepwise=True, trace=True)
    summary = auto_model.summary()

    # 检查模型是否正常
    if is_model_ok(summary):
        print(f"自动选取的模型: {drug_name} + {factory_name}")
        print(summary)
    else:
        print(f"自动选取的模型存在异常: {drug_name} + {factory_name}")
        continue

    # 重置索引，确保索引正确
    log_y = log_y.reset_index(drop=True)
    exog = exog.reset_index(drop=True)
    
    # 进行滚动预测
    log_predictions = []  # 存储对数变换后的预测值
    residuals = []  # 存储残差

    # 使用 SARIMAX 对训练集进行完整预测，获取残差
    for t in range(train_end):
        exog_current = exog.iloc[t: t+1].fillna(0)
        next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
        actual_y_at_t = log_y.iloc[t]
        auto_model.update([actual_y_at_t], exogenous=exog_current)  # 更新 SARIMAX 模型
        residual = actual_y_at_t - next_pred_log  # 计算残差
        residuals.append(residual)

    # 使用历史残差和历史外生变量训练机器学习模型
    shifted_log_y = log_y.shift(1).fillna(0)
    history_exog_with_y = history_exog.copy()
    history_exog_with_y['shifted_log_y'] = shifted_log_y[:train_end]
    ml_model = RandomForestRegressor() # 使用随机森林模型对残差进行预测
    ml_model.fit(history_exog_with_y, residuals)   # 训练机器学习模型
    for t in range(train_end, len(log_y)):      # 使用SARIMAX的预测值加机器学习模型预测残差的混合预测
        # 检查索引是否在范围内，防止超出索引的错误
        if t >= len(exog):
            print(f"超出索引范围：{t}, 最大值为 {len(exog)-1}")
            break
        exog_current = exog.iloc[t: t+1].fillna(0)
        print(exog_current)
        # 使用历史数据进行预测
        try:
            next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
            print(next_pred_log)
        except KeyError as e:
            print(f"索引错误: {e}, 当前 t 值: {t}, exog.iloc[t:t+1] 出错")
        
        # 使用机器学习模型预测残差
        exog_current['shifted_log_y'] = log_y.iloc[t-1]
        print(exog_current)
        if t > train_end:
            predicted_residual = ml_model.predict(exog_current).item()
        else:
            predicted_residual = 0  # 前几期没有足够的数据来预测残差
        
        # 动态调整 scaling_factor
        recent_residuals = residuals[-3:]  # 使用最近3期的残差来动态调整权重
        if len(recent_residuals) > 0:
            scaling_factor = 1 * np.abs(recent_residuals[-1]) / np.mean(np.abs(recent_residuals))
        else:
            scaling_factor = 1
        # 将 residuals 列表临时转换为 Pandas Series 进行 rolling 平滑
        residuals_series = pd.Series(residuals)
        # 将 residuals 列表临时转换为 Pandas Series 进行 rolling 平滑
        if len(residuals) >= 3:
            # 如果已经有足够的历史残差，使用加权平均结合当前预测残差和过去3期残差
            residuals_series = pd.Series(residuals[-3:])  # 取最近3期的残差
            smoothed_residual = 1 * predicted_residual + 0 * residuals_series.mean()  # 给当前预测残差较大的权重
        else:
            smoothed_residual = predicted_residual  # 如果不足3期，使用当前预测残差
        final_pred_log = max(next_pred_log + smoothed_residual * scaling_factor, 0)
        final_pred_log = min(final_pred_log, 1.5 * next_pred_log)
        print(f"Final prediction: {final_pred_log}")
        print(f"Next prediction: {next_pred_log}")
        print(f"Predicted residual: {final_pred_log - next_pred_log}")
        log_predictions.append(final_pred_log)
        # print(log_predictions)
        actual_value = log_y.iloc[t]
        residual = actual_value - next_pred_log  # 计算残差
        residuals.append(residual)

        # 滚动窗口：将真实的观测值加入训练集
        history_y = log_y[:t+1]
        history_exog = exog[:t+1]

        print(f"Residuals shape: {len(residuals)}")
        print(f"History_exog shape: {history_exog.shape}")

        auto_model.update(history_y, exogenous=history_exog)    # 更新 SARIMAX 模型
        history_exog_with_y = history_exog.copy()
        shifted_log_y = log_y.shift(1).fillna(0)
        history_exog_with_y['shifted_log_y'] = shifted_log_y[:t+1]
        ml_model.fit(history_exog_with_y, residuals[-len(history_exog):])  # 更新机器学习模型

    # 还原预测值：从 log 变换还原到原始尺度
    predictions = np.expm1(log_predictions)
    actual_values = np.expm1(log_y[train_end:])  # 还原log后的真实值
    predicted_values = predictions
    # 计算实际值的平均值
    mean_actual = np.mean(actual_values)

    def smape(y_true, y_pred):
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

    # 计算 RMSE
    mae = np.mean(np.abs(actual_values - predicted_values))
    mae_percentage = (mae / mean_actual) * 100
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    smape_value = smape(actual_values, predicted_values)
    
    # 计算 R²
    weights = np.where(actual_values > 0, 1, 0.1)  # 为非零销量天数赋予更高权重
    weighted_r2 = r2_score(actual_values, predicted_values, sample_weight=weights)

    r2_log = r2_score(np.log1p(actual_values), np.log1p(predicted_values))

    # 打印并保存评估结果
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAE%: {mae_percentage}")
    print(f"SMAPE: {smape_value}")
    print(f"R²: {weighted_r2}")
    print(f"R² (log): {r2_log}")

    # 保存模型参数和误差信息
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

    # 保存预测结果
    print(len(predictions))
    print(len(group_data.iloc[train_end:]))

    # 确保 '预测类型' 列存在，如果不存在则创建一个新的空列
    if '预测类型' not in group_data.columns:
        group_data['预测类型'] = ''

    # 确保 '预测减少数量' 列存在，如果不存在则创建一个新的空列
    if '预测减少数量' not in group_data.columns:
        group_data['预测减少数量'] = np.nan  # 初始化为 NaN

    # 使用 .iloc 基于位置的切片，确保 '预测类型' 列有正确的值
    group_data.iloc[train_end:, group_data.columns.get_loc('预测类型')] = '滚动预测'

    # 使用 .iloc 基于位置的切片，确保 '预测减少数量' 列有正确的值
    group_data.iloc[train_end:, group_data.columns.get_loc('预测减少数量')] = pd.Series(predictions).values

    all_results = pd.concat([all_results, group_data])

    # 可视化结果
    plt.figure(figsize=(10, 6))
    # 使用索引进行绘图
    plt.plot(group_data.index, group_data['减少数量'], label='实际减少数量')
    plt.plot(group_data.index[train_end:], predictions, label='滚动预测', linestyle='--')
    
    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('减少数量', fontproperties=font)
    plt.title(f'SARIMAX 滚动预测 vs 实际值 - {drug_name} + {factory_name}', fontproperties=font)
    plt.legend(prop=font)
    # 显示图表3秒
    plt.pause(3)
    # 自动关闭图表
    plt.close()

# 保存模型参数和误差到文件
model_results_df = pd.DataFrame(model_results)
model_results_df.to_csv('model_results.csv', index=False)

# 最后保存所有预测结果到一个文件
all_results.to_csv('all_predicted_results.csv', index=False)
