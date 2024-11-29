import argparse
import pandas as pd
import numpy as np
from prophet import Prophet
from models import config
from models.data_loader import load_data
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

# Define the function to handle outliers
def deal_with_outlier(df):
    """
    Process outliers in the '减少数量' column by grouping by '药品名称' and '厂家'.
    Replace values exceeding 2x or below 0.5x of the group mean with the mean.
    """
    def replace_outliers(group):
        mean_value = group['减少数量'].mean()
        upper_limit = mean_value * 2
        lower_limit = mean_value * 0.5
        group['减少数量'] = np.where(
            (group['减少数量'] > upper_limit) | (group['减少数量'] < lower_limit),
            mean_value,
            group['减少数量']
        )
        return group

    df = df.groupby(['药品名称', '厂家'], group_keys=False).apply(replace_outliers)
    return df

# Command-line arguments
parser = argparse.ArgumentParser(description='Prophet model with rolling predictions')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
parser.add_argument('--initial_train_periods', type=int, default=48, help='Number of initial periods for training')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)
initial_train_periods = args.initial_train_periods

# Load and filter data
df = load_data('updated_monthly_final_combined_df.csv', start_date_filter, end_date_filter)

# Reset index to make 'start_date' a column if it is currently the index
if df.index.name == 'start_date':
    df = df.reset_index()

# Check for the 'start_date' column
if 'start_date' not in df.columns:
    print("Error: 'start_date' column not found. Available columns are:", df.columns)
    exit(1)

# Prepare to store model results
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'rolling_model_results.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

# Process each group (药品名称 + 厂家)
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # Check minimum month requirement
    if len(group_data) < config.min_months:
        print(f"Skipping {drug_name} + {factory_name}: Insufficient data")
        continue

    # Process outliers
    group_data = deal_with_outlier(group_data)

    # Prepare Prophet data format
    group_data = group_data[['start_date', '减少数量']].rename(columns={'start_date': 'ds', '减少数量': 'y'})
    group_data['ds'] = pd.to_datetime(group_data['ds'])  # Ensure date format

    # Rolling prediction with initial training on the first specified periods
    rolling_predictions = []
    actual_values = group_data['y'].values[initial_train_periods:]  # Actual values for future periods to compare
    training_data = group_data.iloc[:initial_train_periods].copy()

    for i in range(initial_train_periods, len(group_data)):
        model = Prophet()
        model.fit(training_data)

        # Predict the next period (1 month ahead)
        future = model.make_future_dataframe(periods=1, freq='ME', include_history=False)
        forecast = model.predict(future)
        next_prediction = forecast['yhat'].values[0]
        
        # Store rolling prediction
        rolling_predictions.append(next_prediction)

        # Append the actual next period's data to the training data
        new_row = group_data.iloc[[i]].copy()
        training_data = pd.concat([training_data, new_row], ignore_index=True)

    # Calculate evaluation metrics
    if len(actual_values) == len(rolling_predictions):
        rmse, mae, smape, r2 = calculate_metrics(actual_values, rolling_predictions)

        # Store results for the current group
        model_result = {
            '药品名称': drug_name, '厂家': factory_name,
            'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
        }
        pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)
        print(f"Results for {drug_name} + {factory_name} have been appended to '{results_file}'")

        # Restore the '减少数量' column name for plotting
        group_data = group_data.rename(columns={'y': '减少数量'})

        # Plot predictions
        plot_predictions(group_data, rolling_predictions, drug_name, factory_name, config.font_path, config.plot_dir)

print("All results have been incrementally saved to 'rolling_model_results.csv'")
