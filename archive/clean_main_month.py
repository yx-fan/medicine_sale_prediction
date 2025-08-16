import os
import pandas as pd
from clean_data.load_files import load_files
from clean_data.process_data import process_data
from clean_data.clean_data import clean_data
from clean_data.aggregate_data import aggregate_data
from clean_data.finalize_data import finalize_data
from clean_data.filter_with_guotan import filter_with_guotan

# Step 1: Load files
folder_path = './dataset'
file_paths = load_files(folder_path)

# Step 2: Process each file and combine data
all_dataframes = []
for file in file_paths:
    # Extract the year from the filename
    try:
        file_name = os.path.basename(file)
        file_year = file_name.split('_')[0].split('.')[0]  # Adjust this as per actual file naming
        if not file_year.isdigit():
            raise ValueError(f"Year extraction failed for file: {file_name}")

        print(f"Processing file: {file}, extracted year: {file_year}")
        
        processed_df = process_data(file, file_year)
        all_dataframes.append(processed_df)

    except ValueError as e:
        print(f"Error: {e}")
        continue

# Combine all DataFrames
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Step 3: Clean data
combined_df = clean_data(combined_df)

# Step 4: Aggregate data
combined_df = aggregate_data(combined_df)

# Step 5: Finalize data
combined_df = finalize_data(combined_df)

# Save to CSV
combined_df.to_csv('final_combined_df.csv', index=False)
print("Data cleaning and processing completed. Output saved to 'final_combined_df.csv'.")

# Step 6: Filter with Guotan
filter_with_guotan()

# Load data
filtered_df = pd.read_csv('filtered_final_combined_df.csv')

# Convert date columns to datetime format
filtered_df['start_date'] = pd.to_datetime(filtered_df['start_date'])
filtered_df['end_date'] = pd.to_datetime(filtered_df['end_date'])

# Group by '药品名称' and '厂家' to retain unique combinations, then resample each group's data to monthly
monthly_dfs = []
for name, group in filtered_df.groupby(['药品名称', '厂家']):
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

# Save the final DataFrame
monthly_df.to_csv('updated_monthly_final_combined_df.csv', index=False)
print("Monthly data has been successfully aggregated and saved to 'updated_monthly_final_combined_df.csv'.")