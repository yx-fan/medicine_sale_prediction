from models.data_loader import load_data
import pandas as pd
from models import config
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import argparse

def advanced_outlier_detection_time_series(df, window_size=7):
    def detect_and_handle_outliers(group):
        # 动态计算滑动窗口均值和标准差
        rolling_mean = group['减少数量'].rolling(window=window_size, min_periods=1).mean()
        rolling_std = group['减少数量'].rolling(window=window_size, min_periods=1).std()

        # 动态计算上下边界
        lower_bound = group['减少数量'].rolling(window=window_size, min_periods=1).quantile(0.05)
        upper_bound = group['减少数量'].rolling(window=window_size, min_periods=1).quantile(0.95)

        # Z-Score 方法检测
        z_scores = np.abs((group['减少数量'] - rolling_mean) / rolling_std)
        outlier_mask = (z_scores > 3) | (group['减少数量'] < lower_bound) | (group['减少数量'] > upper_bound)

        # 将布尔值转换为 0 和 1
        group['is_outlier'] = outlier_mask.astype(int)

        # 修剪异常值到动态上下界
        group.loc[group['减少数量'] < lower_bound, '减少数量'] = lower_bound
        group.loc[group['减少数量'] > upper_bound, '减少数量'] = upper_bound

        return group

    df = (
        df.groupby(['药品名称', '厂家'], group_keys=False)
        .apply(detect_and_handle_outliers)
        .reset_index(drop=True)
    )
    return df

def get_first_nonzero_date(group):
    first_nonzero_date = group.loc[group['减少数量'] > 0, 'start_date'].min()
    group = group[group['start_date'] >= first_nonzero_date]
    return group

parser = argparse.ArgumentParser(description='Advanced SARIMAX Prediction with Enhanced Features')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load data
df = load_data('final_combined_df.csv', start_date_filter, end_date_filter)
df = df.reset_index()
print(df.columns)

# Convert date columns to datetime format
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Group by '药品名称' and '厂家' to retain unique combinations, then resample each group's data to monthly
monthly_dfs = []
for name, group in df.groupby(['药品名称', '厂家']):
    group = group.set_index('start_date').resample('ME').agg({
        '减少数量': 'sum',
        '增加数量': 'sum',
        '期初金额(进价)': 'last',
        '期初金额占比': 'mean'
    }).reset_index()
    group['药品名称'], group['厂家'] = name
    monthly_dfs.append(group)

# Concatenate all grouped monthly data
monthly_df = pd.concat(monthly_dfs, ignore_index=True)

# Calculate 'previous_增加数量' by shifting within each '药品名称' and '厂家' group
monthly_df = monthly_df.sort_values(['药品名称', '厂家', 'start_date'])
monthly_df['previous_增加数量'] = monthly_df.groupby(['药品名称', '厂家'])['增加数量'].shift(1)

# Set each group's start date to the first non-zero data point
monthly_df = (
    monthly_df.groupby(['药品名称', '厂家'], group_keys=False)
    .apply(get_first_nonzero_date)
    .reset_index(drop=True)
)

# Group by '药品名称' and '厂家' to retain unique combinations
unique_groups = monthly_df.groupby(['药品名称', '厂家']).size().reset_index(name='count')

final_df = pd.DataFrame(columns=['药品名称', '厂家', 'start_date', '减少数量', '增加数量', '期初金额(进价)', '期初金额占比', 'previous_增加数量', 'is_outlier'])

reason1 = 0
reason2 = 0
reason3 = 0
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = monthly_df[(monthly_df['药品名称'] == drug_name) & (monthly_df['厂家'] == factory_name)].copy()


    if len(group_data) < config.min_months or group_data['减少数量'].sum() == 0:
        print(f"Skipping {drug_name} + {factory_name}: insufficient or sparse data.")
        reason1 += 1
        del group_data
        continue

    non_zero_ratio = (group_data['减少数量'] != 0).mean()
    if non_zero_ratio < config.sparsity_threshold:
        print(f"Skipping {drug_name} + {factory_name}: Data too sparse (non-zero ratio: {non_zero_ratio:.2f})")
        reason2 += 1
        del group_data
        continue
    
    acf_values = acf(group_data['减少数量'], nlags=12)
    if max(acf_values[1:]) < config.min_acf_threshold:
        print(f"Skipping {drug_name} + {factory_name}: Insufficient autocorrelation.")
        reason3 += 1
        del group_data
        continue

    group_data = advanced_outlier_detection_time_series(group_data)

    print("Final DataFrame columns:", final_df.columns)
    print("Group DataFrame columns:", group_data.columns)
    final_df = pd.concat([final_df, group_data], ignore_index=True)

    del group_data



# Save the final DataFrame
final_df.to_csv('final_monthly_combined_df_after_cleaning.csv', index=False)
print("Data cleaning complete. The cleaned data has been saved to 'final_monthly_combined_df_after_cleaning.csv'.")
print(f"Skipped {reason1} groups due to insufficient or sparse data.")
print(f"Skipped {reason2} groups due to data sparsity.")
print(f"Skipped {reason3} groups due to insufficient autocorrelation.")