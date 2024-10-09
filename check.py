import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

# Load the dataset
file_path = 'filtered_merged_data.csv'
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Check for duplicates
duplicates = data.duplicated(subset=['药品名称', 'week_start_date'])
print(f"Number of duplicated rows: {duplicates.sum()}")
# Check for duplicates after grouping
duplicates = data[data.duplicated(subset=['药品名称', 'week_start_date'], keep=False)]
print("Duplicates based on '药品名称' and 'week_start_date':")
print(duplicates.head(10))  # Inspect the first 10 duplicates

# Check the frequency of the time series (should be weekly and consistent)
data['week_start_date'] = pd.to_datetime(data['week_start_date'])
data = data.sort_values(by=['药品名称', 'week_start_date'])
gaps = data.groupby('药品名称')['week_start_date'].diff().dt.days
print(f"Time gaps between consecutive weeks:\n{gaps.describe()}")

# Visualize the time gaps (if any)
plt.figure(figsize=(10, 5))
gaps.hist(bins=30)
plt.title('Distribution of Time Gaps (Days)')
plt.xlabel('Time Gap (Days)')
plt.ylabel('Frequency')
plt.show()

# Check for variability in '减少数量'
variance_per_product = data.groupby('药品名称')['减少数量'].var()
print(f"Variance of '减少数量' per product:\n{variance_per_product.describe()}")

# Plot a sample product's '减少数量' over time
sample_product = data['药品名称'].unique()[0]  # You can change this to any product name
sample_data = data[data['药品名称'] == sample_product]
plt.figure(figsize=(10, 5))
plt.plot(sample_data['week_start_date'], sample_data['减少数量'])
plt.title(f'{sample_product} - Time Series of 减少数量')
plt.xlabel('Week Start Date')
plt.ylabel('减少数量')
plt.grid(True)
plt.show()

# Stationarity test (ADF Test)
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    return result[1]  # p-value

# Apply ADF test for a sample product
print(f"ADF test for {sample_product}:")
adf_pvalue = adf_test(sample_data['减少数量'])
if adf_pvalue > 0.05:
    print(f"Series is non-stationary (p-value: {adf_pvalue}). Differencing may be needed.")
else:
    print(f"Series is stationary (p-value: {adf_pvalue}).")

# Check for multicollinearity in the exogenous variables (manufacturer columns)
exog_columns = [col for col in data.columns if col.startswith('厂家_')]
exog_corr = data[exog_columns].corr()

# Heatmap of correlations between exogenous variables
plt.figure(figsize=(12, 10))
sns.heatmap(exog_corr, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Exogenous Variables (厂家_ columns)')
plt.show()

# Check for highly correlated columns
high_corr_columns = exog_corr[exog_corr.abs() > 0.8].stack().reset_index()
high_corr_columns = high_corr_columns[high_corr_columns['level_0'] != high_corr_columns['level_1']]  # Exclude self-correlation
if not high_corr_columns.empty:
    print(f"Highly correlated exogenous variables:\n{high_corr_columns}")
else:
    print("No highly correlated exogenous variables found.")

