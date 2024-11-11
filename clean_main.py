import os
import pandas as pd
from clean_data.load_files import load_files
from clean_data.process_data import process_data
from clean_data.clean_data import clean_data
from clean_data.aggregate_data import aggregate_data
from clean_data.finalize_data import finalize_data

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