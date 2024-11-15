import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pmdarima as pm
from models import config
from models.data_loader import load_data
from models.metrics import calculate_metrics

# Define the function to handle outliers
def deal_with_outlier(df):
    def replace_outliers(group):
        mean_value = group['减少数量'].mean()
        upper_limit = mean_value * 2.5
        lower_limit = mean_value * 0.4
        for i in range(len(group)):
            if group['减少数量'].iloc[i] > upper_limit or group['减少数量'].iloc[i] < lower_limit:
                past_3_mean = group['减少数量'].iloc[max(0, i-3):i].mean()
                group.loc[group.index[i], '减少数量'] = past_3_mean if not np.isnan(past_3_mean) else mean_value
        return group

    df = df.groupby(['药品名称', '厂家'], group_keys=False).apply(replace_outliers)
    return df

# SARIMAX model definition with exogenous variables
def train_sarimax_model_month(history_y, history_exog):
    return pm.auto_arima(
        history_y, exogenous=history_exog, seasonal=True, m=12,
        max_d=1, max_p=3, max_q=3, D=1, stepwise=True, trace=True
    )

# LSTM model definition
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Convert data into LSTM format (samples, timesteps, features)
def create_lstm_data(y_data, exog_data, time_steps=1):
    X, y = [], []
    for i in range(time_steps, len(y_data)):
        X.append(np.concatenate([y_data[i-time_steps:i], exog_data[i-time_steps:i]], axis=1))
        y.append(y_data[i])
    return np.array(X), np.array(y)

# Command-line arguments
parser = argparse.ArgumentParser(description='SARIMAX + LSTM model for time series prediction')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load and filter data
df = load_data('updated_monthly_final_combined_df.csv', start_date_filter, end_date_filter)

# Prepare to store model results
unique_groups = df.groupby(['药品名称', '厂家']).size().reset_index(name='count')
results_file = 'model_results.csv'
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

    # Define target variable and exogenous features
    y = group_data['减少数量']
    log_y = np.log1p(y)

    # Extract and merge exogenous variables (previous month’s sales for other manufacturers, 6-month average)
    other_factories = df[(df['药品名称'] == drug_name) & (df['厂家'] != factory_name)]
    for other_factory in other_factories['厂家'].unique():
        other_factory_sales = other_factories[other_factory == other_factories['厂家']].copy()
        other_factory_sales[f'previous_sales_{other_factory}'] = other_factory_sales['减少数量'].shift(1)
        other_factory_sales[f'avg_6_month_sales_{other_factory}'] = (
            other_factory_sales['减少数量']
            .rolling(window=6, min_periods=1)
            .mean()
            .shift(1)
        )
        group_data = pd.merge(
            group_data,
            other_factory_sales[[f'previous_sales_{other_factory}', f'avg_6_month_sales_{other_factory}']],
            left_index=True, right_index=True, how='left'
        )
    group_data.fillna(0, inplace=True)

    # Define exogenous variables
    exog_cols = ['previous_增加数量'] + \
                [f'previous_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()] + \
                [f'avg_6_month_sales_{other_factory}' for other_factory in other_factories['厂家'].unique()]

    exog = group_data[exog_cols]

    # Train SARIMAX model
    history_y = log_y[:config.min_months]
    history_exog = exog[:config.min_months]
    auto_model = train_sarimax_model_month(history_y, history_exog)

    # Use SARIMAX to make initial predictions
    sarimax_predictions = []
    for t in range(config.min_months, len(log_y)):
        exog_current = exog.iloc[t:t + 1].fillna(0)
        next_preds_log = auto_model.predict(n_periods=2, exogenous=exog_current)
        next_pred_log = np.mean(next_preds_log)
        sarimax_predictions.append(next_pred_log)
        actual_y_at_t = log_y.iloc[t]
        auto_model.update([actual_y_at_t], exogenous=exog.iloc[t:t + 1])

    # Convert SARIMAX predictions to original scale
    sarimax_predictions = np.expm1(sarimax_predictions[:len(y) - config.min_months])
    actual_values = np.expm1(log_y[config.min_months:])

    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    exog_scaled = scaler.fit_transform(exog)
    time_steps = 12
    X_lstm, y_lstm = create_lstm_data(y_scaled, exog_scaled, time_steps)

    # Split data into training and testing sets
    train_size = int(len(X_lstm) * 0.8)
    X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
    y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

    # Train LSTM model
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Use LSTM to correct SARIMAX residuals
    lstm_predictions = lstm_model.predict(X_test)
    residuals = scaler.inverse_transform(lstm_predictions)[:, 0]  # Flatten residuals to match SARIMAX shape

    # Align SARIMAX predictions and LSTM residuals length
    sarimax_predictions = sarimax_predictions[-len(residuals):]

    # Evaluate combined model predictions
    combined_predictions = sarimax_predictions + residuals
    rmse, mae, smape, r2 = calculate_metrics(actual_values[-len(combined_predictions):], combined_predictions)
    print(f"Results for {drug_name} + {factory_name}: RMSE={rmse}, MAE={mae}, SMAPE={smape}, R²={r2}")

    # Save results
    model_result = {
        '药品名称': drug_name, '厂家': factory_name,
        'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'R²': r2
    }
    pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)

print("All model results have been saved to 'model_results.csv'")
