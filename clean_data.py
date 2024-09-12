import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the data from Excel file with multiple sheets (tabs)
file_path = 'data.xlsx'  # Replace with your actual file path
all_sheets = pd.read_excel(file_path, sheet_name=None)  # Load all sheets

# Combine all sheets into one DataFrame with an additional 'Month' column
df_list = []
for i, (sheet_name, data) in enumerate(all_sheets.items(), start=1):
    data['Month'] = pd.to_datetime(f'2024-{i:02d}-01')  # Assign a datetime object representing the first day of each month
    df_list.append(data)

combined_df = pd.concat(df_list, ignore_index=True)

# Display the combined DataFrame
print(combined_df.head())

# Clean the data
# Remove rows with missing values in critical columns
combined_df = combined_df.dropna(subset=['药品名称', '药厂', '库存分类', '增加数量'])

# Convert necessary columns to numeric types and handle errors
numeric_columns = ['增加数量', '减少数量', '期初库存', '期末库存']
for col in numeric_columns:
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)

# Convert negative values in '减少数量' to positive, as it's used as sales volume
combined_df['减少数量'] = combined_df['减少数量'].abs()

# Select columns needed for modeling
model_df = combined_df[['药品名称', '药厂', '库存分类', '减少数量', 'Month']]

# Create dummy variables for '药厂' (manufacturer)
model_df = pd.get_dummies(model_df, columns=['药厂'], drop_first=True)

# Only convert columns that are not datetime and not object/string type to float
for col in model_df.columns:
    if model_df[col].dtype not in ['object', 'datetime64[ns]']:
        model_df[col] = model_df[col].astype(float)

# Group data by '药品名称' to create separate datasets for each unique medicine
grouped = model_df.groupby('药品名称')

# Placeholder for SARIMAX model predictions
predictions = {}

print(model_df.head())

# Iterate over each group to build and fit SARIMAX models
for medicine_name, group in grouped:
    print(f'Fitting model for {medicine_name}...')
    print(len(group))
    # Sort the group by 'Month' to ensure chronological order
    group = group.sort_values(by='Month')
    
    # Set 'Month' as index to make it a time series index
    group = group.set_index('Month')
    
    # Separate features (X) and target variable (Y)
    X = group.drop(columns=['减少数量'])  # Ensure X is numeric and valid
    y = group['减少数量']
    
    # Ensure all data in X and y are numeric and check for unexpected object types
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(float)
    
    # Debugging: Print the dtypes of X and Y to ensure they are correct
    print(f'{medicine_name} - X types:\n', X.dtypes)
    print(f'{medicine_name} - Y type:\n', y.dtypes)
    
    # Skip fitting if any object types are detected in X or if Y is not numeric
    if X.select_dtypes(include=['object']).empty and np.issubdtype(y.dtype, np.number):
        # Fit SARIMAX model
        try:
            model = SARIMAX(y, exog=X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            result = model.fit(disp=False)
            # Forecast for the next period (adjust steps as needed)
            forecast = result.forecast(steps=1, exog=X.iloc[-1:])
            predictions[medicine_name] = forecast
            print(f'Model for {medicine_name} fitted successfully.')
        except Exception as e:
            print(f'Error fitting model for {medicine_name}: {e}')
    else:
        print(f'Skipping {medicine_name} due to non-numeric data.')

# Display the predictions
for medicine, forecast in predictions.items():
    print(f'{medicine}: {forecast.values}')
