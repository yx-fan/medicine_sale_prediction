import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import GridSearchCV
import pmdarima as pm
import argparse

sparsity_threshold = 0.2
min_weeks = 65
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')  # 设置中文字体

# 定义超参数网格
#param_grid = {
#    'n_estimators': [100, 200, 300],
#    'max_depth': [5, 10, 20, None],
#    'min_samples_split': [2, 10, 20],
#    'min_samples_leaf': [1, 5, 10],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'bootstrap': [True, False]
#}

df = pd.read_csv('final_combined_df.csv') # 读取数据

# 确保日期格式正确
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.set_index('start_date')
df = df.sort_values('start_date')

unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')  # 获取唯一的 药品名称 + 厂家 组合
model_results = []   # 用于保存模型参数和误差信息的表格
all_results = pd.DataFrame()   # 遍历每个 药品名称 + 厂家 组合

for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    print(f"当前处理的药品名称: {drug_name}, 厂家: {factory_name}")
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()  # 取出该组合的所有数据

    # 如果数据不足 52 周，则跳过
    if len(group_data) < min_weeks:
        print(f"跳过 {drug_name} + {factory_name}，数据不足 {min_weeks} 周")
        continue

    # 如果销量数据过于稀疏，则跳过
    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < sparsity_threshold:
        print(f"跳过 {drug_name} + {factory_name}，销量数据过于稀疏（非零值比例: {non_zero_ratio:.2f}）")
        continue

    # 定义目标变量和外生变量
    y = group_data['减少数量']
    log_y = np.log1p(y)

    # 获取同一药品下的其他药厂数据
    other_factories = df[(df['药品名称'] == drug_name) & (df['厂家'] != factory_name)]
    for other_factory in other_factories['厂家'].unique():   # 遍历每个其他药厂，计算其上一期销量和上六期平均销量
        other_factory_sales = other_factories[other_factories['厂家'] == other_factory].copy()
        other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)
        other_factory_sales[f'avg_6_period_sales_{other_factory}'] = other_factory_sales['减少数量'].rolling(window=6, min_periods=1).mean().shift(1)
        group_data = pd.merge(group_data, other_factory_sales[[f'previous_sales_{other_factory}', f'avg_6_period_sales_{other_factory}']],
                              left_index=True, right_index=True, how='left')
    group_data.fillna(0, inplace=True)  # 填充缺失值

    # 对除期初金额占比之外的外生变量进行log处理
    for other_factory in other_factories['厂家'].unique():
        group_data[f'previous_sales_{other_factory}'] = np.log1p(group_data[f'previous_sales_{other_factory}'])
        group_data[f'avg_6_period_sales_{other_factory}'] = np.log1p(group_data[f'avg_6_period_sales_{other_factory}'])
    group_data['previous_增加数量'] = np.log1p(group_data['previous_增加数量'])

    # 构建外生变量
    exog_cols = ['期初金额占比', 'previous_增加数量'] + [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_period_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()]
    exog = group_data[exog_cols]

    # 初始化模型，使用前 min_weeks 的数据进行初始训练
    train_end = min_weeks
    history_y = log_y[:train_end]  # 使用 log_y
    history_exog = exog[:train_end]
    print(history_y.head())
    print(history_exog.head())

    # 使用 pmdarima 进行自动模型选择
    auto_model = pm.auto_arima(history_y, exogenous=history_exog, seasonal=True, m=52, max_d=2, max_p=2, max_q=2, D=1, stepwise=True, trace=True)
    print(f"自动选取的模型: {drug_name} + {factory_name}")
    print(auto_model.summary())

    # 重置索引，确保索引正确
    log_y = log_y.reset_index(drop=True)
    exog = exog.reset_index(drop=True)
    
    # 进行滚动预测
    log_predictions = []  # 存储对数变换后的预测值
    residuals = []  # 存储残差

    # 使用 SARIMAX 对训练集进行完整预测
    # sarimax_train_pred = auto_model.predict_in_sample(exogenous=history_exog)
    # 计算训练集的残差
    # residuals = list(history_y - sarimax_train_pred)
    # print(f"训练集残差: {residuals}")
    # 首先使用SARIMAX模型对训练集进行预测并计算残差
    # scaler = StandardScaler()
    for t in range(train_end):
        exog_current = exog.iloc[t: t+1].fillna(0)
        next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
    
        actual_value = log_y.iloc[t]
        residual = actual_value - next_pred_log  # 计算残差
        residuals.append(residual)

    # 实例化 GridSearchCV
    #grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
    #                        param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

    # 对历史数据进行调优并训练模型
    #grid_search.fit(history_exog, residuals)

    # 获取最佳模型
    #ml_model = grid_search.best_estimator_
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
            scaling_factor = 0.5 * np.abs(recent_residuals[-1]) / np.mean(np.abs(recent_residuals))
        else:
            scaling_factor = 0.5
        # 将 residuals 列表临时转换为 Pandas Series 进行 rolling 平滑
        residuals_series = pd.Series(residuals)
        # 将 residuals 列表临时转换为 Pandas Series 进行 rolling 平滑
        if len(residuals) >= 3:
            # 如果已经有足够的历史残差，使用加权平均结合当前预测残差和过去3期残差
            residuals_series = pd.Series(residuals[-3:])  # 取最近3期的残差
            smoothed_residual = 1 * predicted_residual + 0 * residuals_series.mean()  # 给当前预测残差较大的权重
        else:
            smoothed_residual = predicted_residual  # 如果不足3期，使用当前预测残差
        print(f"Residuals effect: smoothed_residual * scaling_factor = {smoothed_residual} * {scaling_factor}")
        final_pred_log = next_pred_log + smoothed_residual * scaling_factor
        log_predictions.append(final_pred_log)
        # print(log_predictions)
        actual_value = log_y.iloc[t]
        residual = actual_value - final_pred_log  # 计算残差
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
        ml_model.fit(history_exog_with_y, residuals)  # 更新机器学习模型

    # 还原预测值：从 log 变换还原到原始尺度
    predictions = np.expm1(log_predictions)

    # 计算误差（在还原后的尺度上）
    test_mse = mean_squared_error(np.expm1(log_y[train_end:]), predictions)
    print(f"滚动预测测试集MSE: {test_mse}")

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
        '测试集MSE': test_mse
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
