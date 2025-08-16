import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import configmonth
from models.data_loader import load_data
from models.metrics import calculate_metrics
from models.visualization import plot_predictions

# Command-line arguments
parser = argparse.ArgumentParser(description='Basic XGBoost model with time-based train-test split')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load and filter data
df = load_data('final_monthly_combined_df_after_cleaning.csv', start_date_filter, end_date_filter)
df['药品名称'] = df['药品名称'].str.replace('/', '_')

if df.index.name == 'start_date':
    df = df.reset_index()

df = df.sort_values(by=['药品名称', '厂家', 'start_date'])

# Prepare to store results
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'basic_xgboost_results.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

prediction_file = 'basic_xgboost_predictions.csv'
predictions_columns = ['药品名称', '厂家', 'start_date', 'actual', 'prediction']
pd.DataFrame(columns=predictions_columns).to_csv(prediction_file, index=False)

# Process each group
for _, row in unique_groups.iterrows():
    drug_name = row['药品名称']
    factory_name = row['厂家']
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()

    # Add lag and rolling features
    group_data['ds'] = pd.to_datetime(group_data['start_date'])
    group_data['month'] = group_data['ds'].dt.month
    group_data['quarter'] = group_data['ds'].dt.quarter
    group_data['lag_1'] = group_data['减少数量'].shift(1)
    group_data = group_data.dropna()

    # Define features and target
    X = group_data[['month', 'quarter', 'lag_1']]
    y = group_data['减少数量']

    # Train-test split (first 50% for training, rest for testing)
    split_idx = len(group_data) // 2
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)  # Ensure no negative predictions

    # Evaluate metrics
    rmse, mae, smape, _ = calculate_metrics(y_test.values, predictions)
    r2 = r2_score(y_test, predictions)

    # Save results
    model_result = {
        '药品名称': drug_name,
        '厂家': factory_name,
        'RMSE': rmse,
        'MAE': mae,
        'SMAPE': smape,
        'R²': r2
    }
    pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)

    prediction_results = [
        {
            '药品名称': drug_name,
            '厂家': factory_name,
            'start_date': group_data['ds'].iloc[i],
            'actual': y.iloc[i],
            'prediction': predictions[i - split_idx]
        }
        for i in range(split_idx, len(group_data))
    ]
    pd.DataFrame(prediction_results).to_csv(prediction_file, mode='a', header=False, index=False)

    # Plot predictions
    plot_predictions(group_data.set_index('ds'), predictions, drug_name, factory_name, configmonth.font_path, "model_plots_basic_xgboost")

print("All results have been incrementally saved.")
