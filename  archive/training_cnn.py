import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse

# 设置参数
sparsity_threshold = 0.2
min_weeks = 65
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')

# 读取数据
df = pd.read_csv('filtered_final_combined_df.csv')
df['start_date'] = pd.to_datetime(df['start_date'])
df = df.set_index('start_date').sort_values('start_date')

# 命令行参数解析
parser = argparse.ArgumentParser(description='Train CNN model for time series prediction')
parser.add_argument('--start_date', type=str, required=True, help='The start date for training (format: YYYY-MM-DD)')
args = parser.parse_args()
start_date_filter = pd.to_datetime(args.start_date)
df = df[df.index >= start_date_filter]

unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
model_results = []

# 遍历每个药品名称和厂家组合
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

    y = group_data['减少数量'].values.reshape(-1, 1)

    # 归一化数据
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)

    # 创建序列数据
    def create_sequences(data, sequence_length=10):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    # 生成时间序列
    sequence_length = 10
    X, y = create_sequences(y_scaled, sequence_length)

    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    trainX, testX = X[:split_idx], X[split_idx:]
    trainY, testY = y[:split_idx], y[split_idx:]

    # 定义 CNN 模型
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], 1)),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(trainX, trainY, epochs=50, batch_size=16, verbose=1)

    # 预测结果
    predictions = model.predict(testX)

    # 将预测结果反归一化
    predictions = scaler.inverse_transform(predictions)
    testY_actual = scaler.inverse_transform(testY)

    # 去除 NaN 和负值以避免 log 计算错误
    mask = (testY_actual > 0) & (predictions > 0)
    valid_testY = testY_actual[mask]
    valid_predictions = predictions[mask]

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(testY_actual, predictions))
    mae = mean_absolute_error(testY_actual, predictions)
    mean_actual = np.mean(testY_actual)
    mae_percentage = (mae / mean_actual) * 100
    r2 = r2_score(testY_actual, predictions)
    r2_log = r2_score(np.log1p(valid_testY), np.log1p(valid_predictions)) if len(valid_testY) > 0 else np.nan

    # 输出结果
    print(f"药品名称: {drug_name}, 厂家: {factory_name}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAE%: {mae_percentage}")
    print(f"R²: {r2}")
    print(f"R² (log): {r2_log}")

    # 保存结果
    model_results.append({
        '药品名称': drug_name,
        '厂家': factory_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAE%': mae_percentage,
        'R²': r2,
        'R² (log)': r2_log
    })

    # 绘制预测图表
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(testY_actual)), testY_actual, label='实际值')
    plt.plot(range(len(predictions)), predictions, label='预测值', linestyle='--')
    plt.xlabel('时间步', fontproperties=font)
    plt.ylabel('减少数量', fontproperties=font)
    plt.title(f'CNN 预测 vs 实际值 - {drug_name} + {factory_name}', fontproperties=font)
    plt.legend(prop=font)
    plt.show()

# 保存模型结果到 CSV 文件
model_results_df = pd.DataFrame(model_results)
model_results_df.to_csv('cnn_model_results.csv', index=False)