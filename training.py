import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pmdarima as pm

# 设置中文字体
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

# 读取数据
df = pd.read_csv('final_combined_df.csv')

# 确保日期格式正确
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.sort_values('start_date')

# 获取唯一的 药品名称 + 厂家 组合
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')

# 定义阈值，至少需要 52 周的数据
min_weeks = 65

# 用于保存模型参数和误差信息的表格
model_results = []

# 遍历每个 药品名称 + 厂家 组合
all_results = pd.DataFrame()

for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']

    # 取出该组合的所有数据
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # 如果数据不足 52 周，则跳过
    if len(group_data) < min_weeks:
        print(f"跳过 {drug_name} + {factory_name}，数据不足 {min_weeks} 周")
        continue

    # 定义目标变量和外生变量
    y = group_data['减少数量']
    exog = group_data[['期初金额占比', 'previous_增加数量']]

    print(f"样本量: {len(y[:min_weeks])}")
    # 自动调参
    auto_model = pm.auto_arima(y[:min_weeks], exogenous=exog[:min_weeks], seasonal=True, m=52, max_d=2, D=1, stepwise=True, trace=True)

    # 打印自动选取的模型参数
    print(f"自动选取的模型: {drug_name} + {factory_name}")
    print(auto_model.summary())

    # 预测
    # y_pred_train = auto_model.predict_in_sample(exogenous=exog[:min_weeks])
    if len(group_data) > min_weeks:
        y_pred_test = auto_model.predict(n_periods=len(y) - min_weeks, exogenous=exog[min_weeks:])
    else:
        y_pred_test = None

    # 计算误差
    # train_mse = mean_squared_error(y[:min_weeks], y_pred_train)
    # print(f"训练集MSE: {train_mse}")
    
    if y_pred_test is not None:
        test_mse = mean_squared_error(y[min_weeks:], y_pred_test)
        print(f"测试集MSE: {test_mse}")

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
        # 保存预测结果
    if y_pred_test is not None:
        # 创建一个新的列来标记预测数据
        group_data.loc[min_weeks:, '预测减少数量'] = y_pred_test
        group_data.loc[min_weeks:, '预测类型'] = '测试集预测'

    all_results = pd.concat([all_results, group_data])

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.plot(group_data['start_date'], group_data['减少数量'], label='实际减少数量')
    # plt.plot(group_data['start_date'][:min_weeks], y_pred_train, label='训练集预测', linestyle='--')
    
    if y_pred_test is not None:
        plt.plot(group_data['start_date'][min_weeks:], y_pred_test, label='测试集预测', linestyle='--')
    
    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('减少数量', fontproperties=font)
    plt.title(f'SARIMAX 模型预测 vs 实际值 - {drug_name} + {factory_name}', fontproperties=font)
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
