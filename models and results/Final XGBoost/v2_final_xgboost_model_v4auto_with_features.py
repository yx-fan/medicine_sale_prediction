import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import configmonth
from models.data_loader import load_data
from models.metrics import calculate_metrics
from models.visualization import plot_predictions
import matplotlib.pyplot as plt
import os

# Command-line arguments
parser = argparse.ArgumentParser(description='Rolling XGBoost model with a specific training start date')
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
results_file = 'v2_xgboost_best_model_results_v4.csv'
results_columns = ['药品名称', '厂家', 'RMSE', 'MAE', 'SMAPE', 'R²', 'Best_Params']
pd.DataFrame(columns=results_columns).to_csv(results_file, index=False)

prediction_file = 'v2_xgboost_best_prediction_results_v4.csv'
predictions_columns = ['药品名称', '厂家', 'start_date', 'actual', 'prediction']
pd.DataFrame(columns=predictions_columns).to_csv(prediction_file, index=False)

feature_importance_file = 'v2_xgboost_feature_importance_v4.csv'
feature_importance_columns = ['药品名称', '厂家', 'feature', 'importance', 'rank']
pd.DataFrame(columns=feature_importance_columns).to_csv(feature_importance_file, index=False)

# Parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

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
    group_data['lag_2'] = group_data['减少数量'].shift(2)
    group_data['lag_3'] = group_data['减少数量'].shift(3)
    group_data['rolling_mean_3'] = group_data['减少数量'].shift(1).rolling(window=3).mean()
    group_data = group_data.dropna()

    # Define features and target
    X = group_data[['month', 'quarter', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']]
    y = group_data['减少数量']

    # Initialize variables to track the best model
    initial_train_size = 5
    best_r2 = float('-inf')
    best_params = None
    best_predictions = None
    best_model = None
    r2_results = []

    # Perform grid search
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        params = {
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree,
                            'random_state': 42,
                            'objective': 'reg:squarederror'
                        }

                        rolling_predictions = []
                        rolling_actuals = []

                        for i in range(initial_train_size, len(group_data)):
                            X_train = X.iloc[:i]
                            y_train = y.iloc[:i]
                            X_test = X.iloc[[i]]
                            actual_value = y.iloc[i]

                            model = XGBRegressor(**params)
                            model.fit(X_train, y_train)
                            predicted_value = model.predict(X_test)[0]
                            rolling_predictions.append(max(predicted_value, 0))
                            rolling_actuals.append(actual_value)

                        if rolling_predictions:
                            rolling_predictions = np.array(rolling_predictions)
                            rolling_actuals = np.array(rolling_actuals)
                            valid_indices = ~np.isnan(rolling_actuals) & ~np.isnan(rolling_predictions)
                            if valid_indices.any():
                                r2 = r2_score(
                                    rolling_actuals[valid_indices], rolling_predictions[valid_indices]
                                )
                                r2_results.append({'Params': params, 'R²': r2})
                                if r2 > best_r2:
                                    best_r2 = r2
                                    best_params = params
                                    best_predictions = rolling_predictions

    # Save results
    if best_predictions is not None:
        actual_values = y.iloc[initial_train_size:].values
        rmse, mae, smape, _ = calculate_metrics(actual_values, best_predictions)

        model_result = {
            '药品名称': drug_name,
            '厂家': factory_name,
            'RMSE': rmse,
            'MAE': mae,
            'SMAPE': smape,
            'R²': best_r2,
            'Best_Params': str(best_params)
        }
        pd.DataFrame([model_result]).to_csv(results_file, mode='a', header=False, index=False)

        prediction_results = [
            {
                '药品名称': drug_name,
                '厂家': factory_name,
                'start_date': group_data['ds'].iloc[i],
                'actual': y.iloc[i],
                'prediction': best_predictions[i - initial_train_size]
            }
            for i in range(initial_train_size, len(group_data))
        ]
        pd.DataFrame(prediction_results).to_csv(prediction_file, mode='a', header=False, index=False)

        # Calculate and save feature importance using best model trained on all data
        if best_params is not None:
            # Train final model on all available data (excluding test set) to get feature importance
            final_model = XGBRegressor(**best_params)
            # Use all data except the last few points for feature importance calculation
            train_size = max(initial_train_size, len(X) - 5)
            final_model.fit(X.iloc[:train_size], y.iloc[:train_size])
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance['rank'] = range(1, len(feature_importance) + 1)
            feature_importance['药品名称'] = drug_name
            feature_importance['厂家'] = factory_name
            feature_importance = feature_importance[['药品名称', '厂家', 'feature', 'importance', 'rank']]
            feature_importance.to_csv(feature_importance_file, mode='a', header=False, index=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_importance)), feature_importance['importance'].values)
            plt.yticks(range(len(feature_importance)), feature_importance['feature'].values)
            plt.xlabel('Feature Importance')
            plt.title(f'XGBoost Feature Importance: {drug_name} - {factory_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plot_dir = "model_plots_xgv4auto_v2"
            os.makedirs(plot_dir, exist_ok=True)
            safe_drug_name = drug_name.replace('/', '_').replace('\\', '_')
            safe_factory_name = factory_name.replace('/', '_').replace('\\', '_')
            plt.savefig(f'{plot_dir}/Feature_Importance_{safe_drug_name}_{safe_factory_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Plot predictions
    if best_predictions is not None:
        plot_predictions(group_data.set_index('ds'), best_predictions, drug_name, factory_name, configmonth.font_path, "model_plots_xgv4auto_v2")

# Generate overall feature importance summary
print("Generating overall feature importance summary...")
if os.path.exists(feature_importance_file):
    all_feature_importance = pd.read_csv(feature_importance_file)
    
    # Calculate average importance for each feature across all drug-manufacturer combinations
    overall_importance = all_feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    overall_importance['rank'] = range(1, len(overall_importance) + 1)
    
    # Save overall feature importance
    overall_file = 'v2_xgboost_overall_feature_importance_v4.csv'
    overall_importance.to_csv(overall_file, index=False)
    print(f"Overall feature importance saved to {overall_file}")
    
    # Plot overall feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(overall_importance)), overall_importance['importance'].values)
    plt.yticks(range(len(overall_importance)), overall_importance['feature'].values)
    plt.xlabel('Average Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('XGBoost Overall Feature Importance (Average Across All Drug-Manufacturer Combinations)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_dir = "model_plots_xgv4auto_v2"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/Overall_Feature_Importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overall feature importance plot saved to {plot_dir}/Overall_Feature_Importance.png")
    
    # Print top features
    print("\nTop 5 Most Important Features (Overall):")
    for idx, row in overall_importance.head(5).iterrows():
        print(f"  {row['rank']}. {row['feature']}: {row['importance']:.4f}")

print("All results have been incrementally saved.")
