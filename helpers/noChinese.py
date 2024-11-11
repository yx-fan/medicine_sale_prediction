# Python script to remove Chinese characters in columns E to M across all sheets of an Excel file

import pandas as pd
import re

# Define a function to remove Chinese characters from specified columns
def remove_chinese_characters(df, columns_to_clean):
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: re.sub(r'[\u4e00-\u9fff]', '', str(x)) if pd.notnull(x) else x)
    return df

# Load the Excel file with multiple sheets
file_path = './dataset/2020.01-2020.05_done.xlsx'  # Replace with the actual file path
excel_data = pd.read_excel(file_path, sheet_name=None)  # Load all sheets

# Process each sheet
cleaned_sheets = {}
for sheet_name, sheet_df in excel_data.items():
    columns_to_clean = sheet_df.columns[4:13]  # Select columns from E to M
    cleaned_df = remove_chinese_characters(sheet_df, columns_to_clean)
    cleaned_sheets[sheet_name] = cleaned_df

# Save the cleaned data back to a new Excel file
output_path = './dataset/2020.01-2020.05_done_cleaned_all_sheets.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for sheet_name, cleaned_df in cleaned_sheets.items():
        cleaned_df.to_excel(writer, sheet_name=sheet_name, index=False)

# Define a function to convert specified columns to numeric format
def convert_columns_to_numeric(df, columns_to_convert):
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, set errors as NaN if conversion fails
    return df

# Process each sheet to convert columns E to M to numeric format
formatted_sheets = {}
for sheet_name, sheet_df in excel_data.items():
    columns_to_convert = sheet_df.columns[4:13]  # Columns E to M
    formatted_df = convert_columns_to_numeric(sheet_df, columns_to_convert)
    formatted_sheets[sheet_name] = formatted_df

# Save the formatted data back to a new Excel file
output_path_numeric = './dataset/2020.01-2020.05_done_cleaned_numeric_all_sheets.xlsx'
with pd.ExcelWriter(output_path_numeric) as writer:
    for sheet_name, formatted_df in formatted_sheets.items():
        formatted_df.to_excel(writer, sheet_name=sheet_name, index=False)

output_path_numeric