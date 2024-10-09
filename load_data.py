import pandas as pd
import os

folder_path = './dataset'

# Get all the dicrectories of the files
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith((".xlsx", "xls"))]
print("All files in the dataset: ", file_paths)

# Read all the excel files and store in a list
dfs = [pd.read_excel(file, sheet_name=None) for file in file_paths]
#[
#  {
#    'Sheet1': DataFrame_of_Sheet1_file1,
#    'Sheet2': DataFrame_of_Sheet2_file1,
#    ...
#  },
#  {
#    'Sheet1': DataFrame_of_Sheet1_file2,
#    'Sheet2': DataFrame_of_Sheet2_file2,
#    ...
#  },
#  ...
#]
# Check number of weeks based on the sheets
count = 0
for excel in dfs:
    count += len(excel)
print("Num of weeks in this year: ", count)

# Initialize an empty list to hold all DataFrames
all_dataframes = []
week_num = 1
for file_dict in dfs:
    for sheet_name, df in file_dict.items():
        df["week_num"] = week_num
        all_dataframes.append(df)
        week_num += 1

merged_data = pd.concat(all_dataframes, ignore_index=True)
merged_data = merged_data[["药品名称", "厂家", "减少数量", "week_num"]]

# 1. Attempt to convert '减少数量' to numeric, and report any errors
merged_data['减少数量'] = pd.to_numeric(merged_data['减少数量'], errors='coerce')

# Check for missing values (NaN) in '减少数量' that couldn't be converted to numeric
invalid_entries = merged_data[merged_data['减少数量'].isna()]
if not invalid_entries.empty:
    print("Found invalid entries in '减少数量' that could not be converted to numbers:")
    print(invalid_entries)

# Drop rows with invalid '减少数量' values
merged_data = merged_data.dropna(subset=['减少数量'])

# Convert '减少数量' to absolute integer values after filtering invalid entries
merged_data['减少数量'] = merged_data['减少数量'].abs().astype(int)

# Check for missing values in each column
print("Check missing values: ", merged_data.isnull().sum())
merged_data['厂家'] = merged_data['厂家'].fillna('Unknown')

# 2. Group by '药品名称', 'week_num', and '厂家', aggregating '减少数量'
grouped_data = merged_data.groupby(['药品名称', 'week_num', '厂家'], as_index=False).agg({
    '减少数量': 'sum'
})
print("Group by variables: ", grouped_data.head())

# 3. Convert 'week_num' to a datetime column for time series analysis
start_date = pd.to_datetime('2023-09-01')
grouped_data['week_start_date'] = start_date + pd.to_timedelta(grouped_data['week_num'] - 1, unit='W')

# Debug: Check if 'week_start_date' is in the DataFrame
print("Check if 'week_start_date' exists:", 'week_start_date' in grouped_data.columns)

# 4. One-hot encode the '厂家' column
merged_data_encoded = pd.get_dummies(grouped_data, columns=['厂家'], drop_first=True)
for col in merged_data_encoded.columns:
    if col.startswith('厂家_'):
        merged_data_encoded[col] = merged_data_encoded[col].astype(int)  # 将布尔值转换为整数

# 确保数值列正确设置
merged_data_encoded['减少数量'] = pd.to_numeric(merged_data_encoded['减少数量'], errors='coerce')

# 5. 剔除掉减少数量总和为0的药品
unique_products = merged_data_encoded['药品名称'].unique()
print(f"Total unique products before filtering: {len(unique_products)}")

filtered_dataframes = []
for product in unique_products:
    product_data = merged_data_encoded[merged_data_encoded['药品名称'] == product]
    if product_data['减少数量'].sum() > 0:
        filtered_dataframes.append(product_data)
    else:
        print(f"Skipping {product} because all '减少数量' are 0.")

# 合并保留的产品数据
filtered_merged_data = pd.concat(filtered_dataframes, ignore_index=True)

# 6. 删除所有厂家列中值都为0的列
manufacturer_columns = [col for col in filtered_merged_data.columns if col.startswith('厂家_')]
zero_columns = filtered_merged_data[manufacturer_columns].columns[(filtered_merged_data[manufacturer_columns] == 0).all()]

if len(zero_columns) > 0:
    print(f"Removing the following columns with all zeros: {zero_columns.tolist()}")
    filtered_merged_data.drop(columns=zero_columns, inplace=True)
else:
    print("No columns with all zeros found.")

# 7. 保存处理后的数据到 CSV 文件
filtered_merged_data.to_csv('filtered_merged_data.csv', index=False)

# Debug: 打印最终处理后的数据头部信息
print("Head of the final dataset after filtering:", filtered_merged_data.head())
print(f"Total unique products after filtering: {len(filtered_merged_data['药品名称'].unique())}")