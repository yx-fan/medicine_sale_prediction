import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pmdarima as pm
import argparse

# 设置参数解析器
parser = argparse.ArgumentParser(description='SARIMAX Prediction for Specific Drug and Factory')
parser.add_argument('--drug', type=str, required=True, help='Specify the drug name')
parser.add_argument('--factory', type=str, required=True, help='Specify the factory name')
args = parser.parse_args()

# 获取命令行参数
drug_name_input = args.drug
factory_name_input = args.factory

# 设置中文字体
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

sparsity_threshold = 0.2

# 读取数据
df = pd.read_csv('final_combined_df.csv')

# 确保日期格式正确
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.set_index('start_date')
df = df.sort_values('start_date')

# 定义阈值，至少需要 52 周的数据
min_weeks = 65

# 取出指定药品名称和厂家组合的数据
group_data = df[(df['药品名称'] == drug_name_input) & (df['厂家'] == factory_name_input)].copy()

# 如果数据不足 52 周，则跳过
if len(group_data) < min_weeks:
    print(f"跳过 {drug_name_input} + {factory_name_input}，数据不足 {min_weeks} 周")
else:
    # 计算非零值比例
    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < sparsity_threshold:
        print(f"跳过 {drug_name_input} + {factory_name_input}，销量数据过于稀疏（非零值比例: {non_zero_ratio:.2f}）")
    else:
        # 定义目标变量和外生变量
        y = group_data['减少数量']

        # 将 y 进行对数变换
        log_y = np.log1p(y)

        # 获取同一药品下的其他药厂数据
        other_factories = df[(df['药品名称'] == drug_name_input) & (df['厂家'] != factory_name_input)]

        # 遍历每个其他药厂，计算其上一期销量和上六期平均销量
        for other_factory in other_factories['厂家'].unique():
            # 获取该药厂的销量
            other_factory_sales = other_factories[other_factories['厂家'] == other_factory].copy()

            # 计算上一期的销量
            other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)

            # 计算上六期的平均销量
            other_factory_sales[f'avg_6_period_sales_{other_factory}'] = other_factory_sales['减少数量'].rolling(window=6, min_periods=1).mean().shift(1)

            # 合并其他药厂的上一期销量和上六期平均销量
            group_data = pd.merge(group_data, other_factory_sales[[f'previous_sales_{other_factory}', f'avg_6_period_sales_{other_factory}']],
                                  left_index=True, right_index=True, how='left')

        # 填充缺失值
        group_data.fillna(0, inplace=True)

        # 对外生变量进行log处理
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

        # 使用 pmdarima 进行自动模型选择
        auto_model = pm.auto_arima(history_y, exogenous=history_exog, seasonal=True, m=52, max_d=2, max_p=2, max_q=2, D=1, stepwise=True, trace=True)

        print(f"自动选取的模型: {drug_name_input} + {factory_name_input}")
        print(auto_model.summary())

        log_y = log_y.reset_index(drop=True)
        exog = exog.reset_index(drop=True)

        # 进行滚动预测
        log_predictions = []  # 存储对数变换后的预测值
        for t in range(train_end, len(log_y)):
            # 检查索引是否在范围内，防止超出索引的错误
            if t >= len(exog):
                print(f"超出索引范围：{t}, 最大值为 {len(exog)-1}")
                break
            exog_current = exog.iloc[t: t+1].fillna(0)
            print(f"当前 exog 值: {exog_current}")
            if exog_current.isnull().values.any():
                print(f"检测到 NaN 值，t={t}")
            # 使用历史数据进行预测
            try:
                next_pred_log = auto_model.predict(n_periods=1, exogenous=exog_current).item()
                print(next_pred_log)
            except KeyError as e:
                print(f"索引错误: {e}, 当前 t 值: {t}, exog.iloc[t:t+1] 出错")
            log_predictions.append(next_pred_log)

        # 还原预测值：从 log 变换还原到原始尺度
        predictions = np.expm1(log_predictions)

        # 计算误差
        test_mse = mean_squared_error(np.expm1(log_y[train_end:]), predictions)
        print(f"滚动预测测试集MSE: {test_mse}")

        # 可视化结果
        plt.figure(figsize=(10, 6))
        # 使用索引进行绘图
        plt.plot(group_data.index, group_data['减少数量'], label='实际减少数量')
        plt.plot(group_data.index[train_end:], predictions, label='滚动预测', linestyle='--')
        plt.xlabel('日期', fontproperties=font)
        plt.ylabel('减少数量', fontproperties=font)
        plt.title(f'SARIMAX 滚动预测 vs 实际值 - {drug_name_input} + {factory_name_input}', fontproperties=font)
        plt.legend(prop=font)
        plt.show()
