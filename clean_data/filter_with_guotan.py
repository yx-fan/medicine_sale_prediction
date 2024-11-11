import pandas as pd
import os

def filter_with_guotan():
    # Define paths to the CSV files
    main_df_path = 'final_combined_df.csv'  # Root directory
    guotanyaopin_df_path = os.path.join('clean_data', 'guotanyaopin.csv')  # Inside clean_data folder

    # Load the CSV files
    main_df = pd.read_csv(main_df_path)
    guotanyaopin_df = pd.read_csv(guotanyaopin_df_path)

    # Combine 药品名称 and 厂家 columns in both dataframes for matching
    guotanyaopin_df['药品_厂家'] = guotanyaopin_df['药品名称'] + '_' + guotanyaopin_df['厂家']
    main_df['药品_厂家'] = main_df['药品名称'] + '_' + main_df['厂家']

    # Filter main_df to keep only rows where 药品_厂家 exists in guotanyaopin_df
    filtered_main_df = main_df[main_df['药品_厂家'].isin(guotanyaopin_df['药品_厂家'])]

    # Save the filtered data for further modeling
    filtered_main_df.to_csv('filtered_final_combined_df.csv', index=False)
    print("Filtered data saved to 'filtered_final_combined_df.csv'")

