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
from joblib import Parallel, delayed
from itertools import product

# Check if GPU is available
def check_gpu_available():
    """Check if CUDA GPU is available for XGBoost"""
    try:
        # First check if nvidia-smi is available
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # GPU hardware exists, now check if XGBoost supports it
            try:
                # Try to import and check XGBoost GPU support
                import xgboost as xgb
                # Check if GPU is available by trying to create a small test
                test_data = np.array([[1, 2], [3, 4]])
                test_label = np.array([1, 2])
                try:
                    # This will fail if GPU is not properly configured
                    test_model = XGBRegressor(
                        tree_method='hist',  # Use hist method with GPU
                        device='cuda',  # Use CUDA device
                        n_estimators=1,
                        max_depth=1
                    )
                    test_model.fit(test_data, test_label)
                    return True
                except Exception as e:
                    # GPU exists but XGBoost can't use it (likely missing CUDA support)
                    print(f"[INFO] GPU detected but XGBoost GPU support not available: {e}")
                    return False
            except Exception as e:
                return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi not found or timeout
        pass
    except Exception as e:
        pass
    return False

# Detect GPU availability
USE_GPU = check_gpu_available()
if USE_GPU:
    print("[INFO] ✓ NVIDIA GPU detected! Will use GPU to reduce CPU temperature and load.")
else:
    print("[INFO] No GPU detected or GPU not available. Using CPU (may run hotter with high load).")

# Command-line arguments
parser = argparse.ArgumentParser(description='Rolling XGBoost model with a specific training start date')
parser.add_argument('--start_date', type=str, required=True, help='Start date for training (YYYY-MM-DD)')
parser.add_argument('--end_date', type=str, required=True, help='End date for training (YYYY-MM-DD)')
args = parser.parse_args()

start_date_filter = pd.to_datetime(args.start_date)
end_date_filter = pd.to_datetime(args.end_date)

# Load and filter data
print(f"[INFO] Loading data from final_monthly_combined_df_after_cleaning.csv...")
print(f"[INFO] Date range: {start_date_filter} to {end_date_filter}")
df = load_data('final_monthly_combined_df_after_cleaning.csv', start_date_filter, end_date_filter)
print(f"[INFO] Loaded {len(df)} rows of data")
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
total_groups = len(unique_groups)
print(f"\n[INFO] Starting to process {total_groups} drug-manufacturer combinations...")
print("=" * 80)

for idx, row in enumerate(unique_groups.iterrows(), 1):
    drug_name = row[1]['药品名称']
    factory_name = row[1]['厂家']
    print(f"\n[{idx}/{total_groups}] Processing: {drug_name} - {factory_name}")
    group_data = df[(df['药品名称'] == drug_name) & (df['厂家'] == factory_name)].copy()
    print(f"  - Data points: {len(group_data)}")

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

    # Calculate total parameter combinations
    total_combinations = (len(param_grid['n_estimators']) * 
                         len(param_grid['learning_rate']) * 
                         len(param_grid['max_depth']) * 
                         len(param_grid['subsample']) * 
                         len(param_grid['colsample_bytree']))
    print(f"  - Starting grid search with {total_combinations} parameter combinations...")
    
    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid['n_estimators'],
        param_grid['learning_rate'],
        param_grid['max_depth'],
        param_grid['subsample'],
        param_grid['colsample_bytree']
    ))
    
    def evaluate_params(params_tuple):
        """Evaluate a single parameter combination"""
        n_estimators, learning_rate, max_depth, subsample, colsample_bytree = params_tuple
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'n_jobs': 1,  # Use 1 thread per model since we're parallelizing at parameter level
        }
        
        # Use GPU only for large datasets to avoid memory issues
        # For small datasets (< 10000 samples), GPU causes OOM with parallel processing
        data_size = len(group_data)
        if USE_GPU and data_size > 10000:
            params['tree_method'] = 'hist'  # Use histogram algorithm
            params['device'] = 'cuda'  # Use CUDA device (this enables GPU acceleration)
        
        rolling_predictions = []
        rolling_actuals = []
        
        # Rolling window training - this is the time-consuming part
        num_rolling_steps = len(group_data) - initial_train_size
        for step, i in enumerate(range(initial_train_size, len(group_data)), 1):
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
                return {'Params': params, 'R²': r2, 'predictions': rolling_predictions}
        return None
    
    # Parallel grid search - leave some CPU cores for other programs
    import multiprocessing
    import time
    total_cores = multiprocessing.cpu_count()
    # For small datasets, GPU is not beneficial and causes memory issues with parallel processing
    # GPU is only useful for large datasets (millions of samples)
    # With parallel processing, each process uses GPU memory, causing OOM
    data_size = len(group_data)
    use_gpu_for_this = USE_GPU and data_size > 10000  # Only use GPU for large datasets
    
    if use_gpu_for_this:
        # For large datasets with GPU: use very few parallel processes to avoid GPU memory issues
        n_jobs = max(2, min(4, int(total_cores * 0.25)))  # Much fewer processes to avoid GPU OOM
        print(f"  - Using GPU acceleration + {n_jobs}/{total_cores} CPU cores (limited parallelism to avoid GPU memory issues)...")
    else:
        if USE_GPU and data_size <= 10000:
            print(f"  - Dataset too small ({data_size} samples) for GPU benefits. Using CPU instead (GPU causes memory issues with parallel processing).")
        # Use CPU: can use more parallel processes
        n_jobs = max(2, min(12, int(total_cores * 0.75)))
        print(f"  - Using parallel processing with {n_jobs}/{total_cores} CPU cores (leaving {total_cores - n_jobs} cores for other programs)...")
    
    print(f"  - Estimated time: ~{total_combinations * (len(group_data) - initial_train_size) / n_jobs / 10:.1f} seconds per combination (rough estimate)")
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(evaluate_params)(params_tuple) for params_tuple in param_combinations
    )
    elapsed_time = time.time() - start_time
    print(f"  - Grid search completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    # Find best result
    for result in results:
        if result is not None:
            r2_results.append({'Params': result['Params'], 'R²': result['R²']})
            if result['R²'] > best_r2:
                best_r2 = result['R²']
                best_params = result['Params']
                best_predictions = result['predictions']
                print(f"    - New best R²: {best_r2:.4f}")
    
    print(f"  - Grid search completed. Best R²: {best_r2:.4f}")

    # Save results
    if best_predictions is not None:
        print(f"  - Calculating metrics and saving results...")
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
        print(f"  - Results saved: RMSE={rmse:.2f}, MAE={mae:.2f}, SMAPE={smape:.2f}%, R²={best_r2:.4f}")

        # Calculate and save feature importance using best model trained on all data
        if best_params is not None:
            print(f"  - Calculating feature importance...")
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
            # Use English labels to avoid font issues
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_importance)), feature_importance['importance'].values)
            plt.yticks(range(len(feature_importance)), feature_importance['feature'].values)
            plt.xlabel('Feature Importance', fontsize=12)
            # Use safe title without Chinese characters to avoid font issues
            safe_title = f'XGBoost Feature Importance: {drug_name[:20]} - {factory_name[:20]}'
            plt.title(safe_title, fontsize=12)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plot_dir = "model_plots_xgv4auto_v2"
            os.makedirs(plot_dir, exist_ok=True)
            safe_drug_name = drug_name.replace('/', '_').replace('\\', '_')
            safe_factory_name = factory_name.replace('/', '_').replace('\\', '_')
            plt.savefig(f'{plot_dir}/Feature_Importance_{safe_drug_name}_{safe_factory_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  - Feature importance plot saved")

    # Plot predictions
    if best_predictions is not None:
        print(f"  - Generating prediction plot...")
        # Use None for font_path, let the function auto-detect
        try:
            plot_predictions(group_data.set_index('ds'), best_predictions, drug_name, factory_name, None, "model_plots_xgv4auto_v2")
        except Exception as e:
            print(f"  - Warning: Could not generate prediction plot: {e}")
            print(f"  - Continuing without plot...")
        print(f"  ✓ Completed: {drug_name} - {factory_name}")
    else:
        print(f"  ✗ Skipped: {drug_name} - {factory_name} (insufficient data)")

# Generate overall feature importance summary
print("\n" + "=" * 80)
print("[INFO] Generating overall feature importance summary...")
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

    print("\n" + "=" * 80)
    print("[INFO] All processing completed!")
    print(f"[INFO] Results saved to: {results_file}")
    print(f"[INFO] Predictions saved to: {prediction_file}")
    print(f"[INFO] Feature importance saved to: {feature_importance_file}")
else:
    print("\n[WARNING] No feature importance data found. Skipping overall summary.")
    print("[INFO] All results have been incrementally saved.")
