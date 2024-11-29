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

df = pd.read_csv('final_combined_df.csv') # 读取数据

# 确保日期格式正确
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.set_index('start_date')
df = df.sort_values('start_date')

# 根据指定的start_date进行过滤
df = df[df.index >= start_date_filter]
print(f"训练数据从 {start_date_filter} 开始")

unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')  # 获取唯一的 药品名称 + 厂家 组合
model_results = []   # 用于保存模型参数和误差信息的表格
all_results = pd.DataFrame()   # 遍历每个 药品名称 + 厂家 组合

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
    if non_zero_ratio < sparsity_threshold:
        print(f"跳过 {drug_name} + {factory_name}，销量数据过于稀疏（非零值比例: {non_zero_ratio:.2f}）")
        continue

    # 定义目标变量和外生变量
    y = group_data['减少数量']
    log_y = np.log1p(y)

    # 获取同一药品下的其他药厂数据
    other_factories = df[(df['药品名称'] == drug_name) & (df['厂家'] != factory_name)]
    for other_factory in other_factories['厂家'].unique():  # 遍历每个其他药厂，计算其上一期销量和上六期平均销量
        other_factory_sales = other_factories[other_factories['厂家'] == other_factory].copy()
        other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)
        other_factory_sales[f'avg_6_period_sales_{other_factory}'] = other_factory_sales['减少数量'].rolling(window=6,
                                                                                                         min_periods=1).mean().shift(
            1)
        group_data = pd.merge(group_data, other_factory_sales[
            [f'previous_sales_{other_factory}', f'avg_6_period_sales_{other_factory}']],
                              left_index=True, right_index=True, how='left')
    group_data.fillna(0, inplace=True)  # 填充缺失值

    # 对外生变量进行 log 处理
    for other_factory in other_factories['厂家'].unique():
        group_data[f'previous_sales_{other_factory}'] = np.log1p(group_data[f'previous_sales_{other_factory}'])
        group_data[f'avg_6_period_sales_{other_factory}'] = np.log1p(group_data[f'avg_6_period_sales_{other_factory}'])

    # 构建外生变量
    exog_cols = [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_period_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()]
    exog = group_data[exog_cols]

    # 使用初始训练集数据来训练和预测
    train_end = min_weeks
    history_y = log_y[:train_end]  # 使用 log_y
    history_exog = exog[:train_end]
    print(history_y.head())
    print(history_exog.head())

    # 使用 pmdarima 进行自动模型选择
    auto_model = pm.auto_arima(history_y, exogenous=history_exog, seasonal=True, m=52, max_d=2, max_p=3, max_q=3, D=1,
                               stepwise=True, trace=True)
    summary = auto_model.summary()

    if is_model_ok(summary):
        print(f"自动选取的模型: {drug_name} + {factory_name}")
        print(summary)
    else:
        print(f"自动选取的模型存在异常: {drug_name} + {factory_name}")
        continue

    # 进行单次预测（而非滚动预测）
    exog_test = exog[train_end:]
    predicted_log_values = auto_model.predict(n_periods=len(exog_test), exogenous=exog_test)

    # 还原预测值：从 log 变换还原到原始尺度
    predictions = np.expm1(predicted_log_values)
    actual_values = np.expm1(log_y[train_end:])  # 还原 log 后的真实值
    predicted_values = predictions

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(group_data.index, group_data['减少数量'], label='实际减少数量')
    plt.plot(group_data.index[train_end:], predictions, label='预测', linestyle='--')
    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('减少数量', fontproperties=font)
    plt.title(f'SARIMAX 预测 vs 实际值 - {drug_name} + {factory_name}', fontproperties=font)
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
