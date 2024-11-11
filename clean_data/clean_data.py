import pandas as pd

def clean_data(df):
    df['厂家'] = df['厂家'].fillna('Unknown')
    df['期初金额(进价)'] = pd.to_numeric(df['期初金额(进价)'], errors='coerce')
    df['增加数量'] = pd.to_numeric(df['增加数量'], errors='coerce')
    df['减少数量'] = pd.to_numeric(df['减少数量'], errors='coerce').abs()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    return df