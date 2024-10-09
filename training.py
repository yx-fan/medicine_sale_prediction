import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 加载数据并确保 'week_start_date' 为 datetime 格式
file_path = './merged_data.csv'
merged_data = pd.read_csv(file_path)

# 确保 'week_start_date' 是 datetime 格式
merged_data['week_start_date'] = pd.to_datetime(merged_data['week_start_date'])

# 初始数据预览和总行数检查
print("Initial data preview:\n", merged_data.head())
print("Total rows after loading:", len(merged_data))

# 检查并打印不同药品的分布情况
print("Unique products (药品名称):", merged_data['药品名称'].unique())
print("Product count distribution:\n", merged_data['药品名称'].value_counts())

# 检查是否有重复的记录
duplicates_before = merged_data[merged_data.duplicated(subset=['药品名称', 'week_start_date'], keep=False)]
print(f"Before aggregation, found {len(duplicates_before)} duplicated rows")

# 对相同的 '药品名称' 和 'week_start_date' 进行聚合，确保每个时间点只有一条数据
merged_data = merged_data.groupby(['药品名称', 'week_start_date']).agg({
    '减少数量': 'sum',
    **{col: 'first' for col in merged_data.columns if col.startswith('厂家_')}
}).reset_index()

# 再次检查是否还有重复的时间索引
duplicates_after = merged_data[merged_data.duplicated(subset=['药品名称', 'week_start_date'], keep=False)]
print(f"After aggregation, found {len(duplicates_after)} duplicated rows")

# 打印聚合后数据的基本信息
print("After aggregation, total rows:", len(merged_data))
print("Unique products after aggregation:", merged_data['药品名称'].unique())

# 检查是否仍然有大量的重复索引，并解决这个问题
merged_data.set_index('week_start_date', inplace=True)
if merged_data.index.duplicated().sum() > 0:
    print(f"Found {merged_data.index.duplicated().sum()} duplicated index entries. Removing duplicates.")
    merged_data = merged_data.loc[~merged_data.index.duplicated()]

# 确保 '减少数量' 列为数值型并处理缺失值
merged_data['减少数量'] = pd.to_numeric(merged_data['减少数量'], errors='coerce').fillna(0)

# 确保厂家相关的列为数值型，将布尔值转换为整数
exog_columns = [col for col in merged_data.columns if col.startswith("厂家_")]
for col in exog_columns:
    merged_data[col] = merged_data[col].astype(int)

# 打印列的数据类型，以确保没有布尔值或其他非数值类型
print("Data types after numeric conversion:\n", merged_data.dtypes)

# 获取所有产品名称
products = merged_data['药品名称'].unique()
print("Total unique products after cleaning: ", len(products))

# 初始化预测结果列表
forecasts = []
skipped_products = 0

# 遍历每个产品并进行 SARIMAX 预测
for product in products:
    print(f"Processing {product}...")

    # 过滤当前产品的数据
    product_data = merged_data[merged_data["药品名称"] == product]

    # 打印减少数量的总和
    total_sales = product_data['减少数量'].sum()
    print(f"Total '减少数量' for {product}: {total_sales}")

    # 检查是否所有的 '减少数量' 都是 0
    if total_sales == 0:
        print(f"Skipping {product} because all '减少数量' are 0.")
        skipped_products += 1
        continue

    # 按索引排序
    product_data = product_data.sort_index()

    # 根据日期进行训练集和测试集的划分
    split_date = product_data.index[int(len(product_data) * 0.8)]  # 80% 用于训练
    train_data = product_data.loc[:split_date]
    test_data = product_data.loc[split_date:]

    # 厂家相关的列
    exog_columns = [col for col in product_data.columns if col.startswith("厂家_")]

    # 分割目标变量 (减少数量) 和外生变量 (厂家)
    train_target = train_data['减少数量']
    test_target = test_data['减少数量']
    train_exog = train_data[exog_columns]
    test_exog = test_data[exog_columns]

    # 打印外生变量的情况
    print(f"Train exogenous variables for {product}:\n", train_exog.head())
    print(f"Exogenous variable types for {product}:\n", train_exog.dtypes)

    try:
        # 建立并拟合 SARIMAX 模型
        model = SARIMAX(train_target, exog=train_exog, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
        sarimax_result = model.fit(disp=False)

        # 预测测试期
        forecast = sarimax_result.forecast(steps=len(test_target), exog=test_exog)

        # 保存预测结果
        forecasts.append({
            '药品名称': product,
            'week_start_date': test_data.index.values,
            'actual': test_target.values,
            'forecast': forecast.values
        })

        # 绘制实际值与预测值的对比
        plt.figure(figsize=(10,6))
        plt.plot(train_data.index, train_target, label='Training Data')
        plt.plot(test_data.index, test_target, label='Actual Sales')
        plt.plot(test_data.index, forecast, label='Forecasted Sales', linestyle='--')
        plt.title(f'SARIMAX Forecast for {product}')
        plt.xlabel('Week Start Date')
        plt.ylabel('Sales (减少数量)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 计算评估指标
        mae = mean_absolute_error(test_target, forecast)
        mse = mean_squared_error(test_target, forecast)
        print(f"Product: {product} - MAE: {mae}, MSE: {mse}")

    except Exception as e:
        print(f"Error processing {product}: {e}")

print(f"Total products skipped: {skipped_products}")
