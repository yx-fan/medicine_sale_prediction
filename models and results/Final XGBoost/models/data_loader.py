import pandas as pd

def load_data(file_path, start_date, end_date):
    df = pd.read_csv(file_path)
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.set_index('start_date').sort_values('start_date')
    return df[(df.index >= start_date) & (df.index <= end_date)]