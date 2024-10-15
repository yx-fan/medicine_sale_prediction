import pandas as pd
import os
from datetime import datetime
import numpy as np

# Function to parse the date range string and handle cases where the month crosses over
def parse_week_range(week_range, file_year):
    start_str, end_str = week_range.split('-')  # Split the week range

    # 解析 start 和 end 的日期（start 和 end 可以是类似 "1.2" 或 "2"）
    start_month_day = start_str.split('.')  # Split month and day of the start date
    if '.' in end_str:  # If the end date contains a month as well
        end_month_day = end_str.split('.')
    else:
        end_month_day = [start_month_day[0], end_str]  # Use the same month for the end date
    
    start_month, start_day = int(start_month_day[0]), int(start_month_day[1])
    end_month, end_day = int(end_month_day[0]), int(end_month_day[1])

    # Handle year wrapping around December to January
    start_year = int(file_year)
    if start_month > end_month:
        end_year = start_year + 1  # 跨年处理
    else:
        end_year = start_year

    # Create start and end date objects
    start_date = datetime(year=start_year, month=start_month, day=start_day)
    end_date = datetime(year=end_year, month=end_month, day=end_day)
    
    return start_date, end_date

# Sample folder path
folder_path = './dataset'

# Get all the directories of the files
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith((".xlsx", "xls"))]

# Sort the file paths based on the date in the filename
file_paths.sort(key=lambda x: os.path.basename(x).split('_')[0])

all_dataframes = []

for file in file_paths:
    # Extract the year from the filename (assuming format like '2024.01_done.xlsx')
    file_year = os.path.basename(file).split('_')[0].split('.')[0]  # Extract '2024' from '2024.01_done.xlsx'
    
    # Read the Excel file
    file_dict = pd.read_excel(file, sheet_name=None)
    
    for sheet_name, df in file_dict.items():
        if '药厂' in df.columns:
            df = df.rename(columns={'药厂': '厂家'})
            print("rename")

        # Parse the week range from the sheet name (e.g., '1.2-1.8')
        week_range = sheet_name
        print(f"Processing sheet: {week_range}")
        
        # Get start and end dates for the week
        start_date, end_date = parse_week_range(week_range, file_year)
        print(f"Parsed start_date: {start_date}, end_date: {end_date}")
        
        # Select specific columns (replace 'col1', 'col2', etc. with actual column names)
        selected_columns = ['药品名称', '厂家', '增加数量', '减少数量', '期初金额(进价)']  # Adjust this to your required columns
        if all(col in df.columns for col in selected_columns):
            df = df.loc[:, selected_columns]  # Use .loc to safely select the columns
            
            # Add the start and end dates to the DataFrame safely using .loc
            df.loc[:, 'start_date'] = start_date
            df.loc[:, 'end_date'] = end_date
            
            # Append the DataFrame to the list
            all_dataframes.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(all_dataframes, ignore_index=True)

combined_df.head(20000).to_csv('combined_df_first_20000.csv', index=False)

# Show the combined DataFrame with the week range dates
print(combined_df.head())

missing_data = combined_df.isnull().sum()
print("含有缺失值的列：")
print(missing_data[missing_data > 0])
combined_df['厂家'] = combined_df['厂家'].fillna('Unknown')
combined_df['期初金额(进价)'] = pd.to_numeric(combined_df['期初金额(进价)'], errors='coerce')
combined_df['增加数量'] = pd.to_numeric(combined_df['增加数量'], errors='coerce')
combined_df['减少数量'] = pd.to_numeric(combined_df['减少数量'], errors='coerce').abs()
combined_df['start_date'] = pd.to_datetime(combined_df['start_date'])
combined_df['end_date'] = pd.to_datetime(combined_df['end_date'])

combined_df = combined_df.groupby(['药品名称', '厂家', 'start_date', 'end_date'], as_index=False).agg({
    '减少数量': 'sum',
    '增加数量': 'sum',
    '期初金额(进价)': 'sum'
})

combined_df['减少数量总和'] = combined_df.groupby(['药品名称', '厂家'])['减少数量'].transform('sum')
combined_df = combined_df[combined_df['减少数量总和'] != 0].copy()
combined_df.drop(columns=['减少数量总和'], inplace=True)

duplicate_rows = combined_df[combined_df.duplicated()]
print(f"重复行的数量: {duplicate_rows.shape[0]}")
print(combined_df.head())
duplicate_count = combined_df.duplicated(subset=['药品名称', '厂家', 'start_date']).sum()
print(f"重复的组合数量: {duplicate_count}")

# Group by 药品名称 and start_date to calculate total 期初金额 for each 药品名称 per period
combined_df['total_期初金额_by_药品名称'] = combined_df.groupby(['药品名称', 'start_date'])['期初金额(进价)'].transform('sum')

# Calculate the proportion of each 药品名称 + 药厂's 期初金额 to the total
combined_df['期初金额占比'] = np.where(combined_df['期初金额(进价)'] == 0, 0, combined_df['期初金额(进价)'] / combined_df['total_期初金额_by_药品名称'])
combined_df = combined_df.sort_values(by=['药品名称', '厂家', 'start_date'])
combined_df['previous_增加数量'] = combined_df.groupby(['药品名称', '厂家'])['增加数量'].shift(1)
combined_df['previous_增加数量'] = combined_df['previous_增加数量'].fillna(0)
combined_df.drop(columns=['total_期初金额_by_药品名称'], inplace=True)

print(combined_df.head())


# 按 '药品名称', '厂家' 进行分组，并检查每个组内的日期间隔
for name, group in combined_df.groupby(['药品名称', '厂家']):
    # 先按日期排序，确保日期顺序正确
    group = group.sort_values(by='start_date')
    
    # 计算每个组的日期差异
    date_diff = group['start_date'].diff().dropna()
    
    # 打印该组的日期间隔，如果有多个不同的间隔，说明有问题
    unique_diff = date_diff.unique()
    
    # 如果不全是期望的间隔（例如7天），输出该组的行
    if len(unique_diff) > 1 or unique_diff[0] != pd.Timedelta(days=7):
        print(f"存在不一致的日期间隔, 组名: {name}")
        print(group[['药品名称', '厂家', 'start_date', 'end_date']])
        print(unique_diff)


combined_df.to_csv('final_combined_df.csv', index=False)