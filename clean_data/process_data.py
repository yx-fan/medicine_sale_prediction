import pandas as pd
from .parse_week_range import parse_week_range

def process_data(file, file_year):
    file_dict = pd.read_excel(file, sheet_name=None)
    all_dataframes = []
    
    for sheet_name, df in file_dict.items():
        if '药厂' in df.columns:
            df = df.rename(columns={'药厂': '厂家'})
        
        week_range = sheet_name
        start_date, end_date = parse_week_range(week_range, file_year)
        
        selected_columns = ['药品名称', '厂家', '增加数量', '减少数量', '期初金额(进价)']
        if all(col in df.columns for col in selected_columns):
            df = df.loc[:, selected_columns]
            df['start_date'] = start_date
            df['end_date'] = end_date
            all_dataframes.append(df)
    
    return pd.concat(all_dataframes, ignore_index=True)