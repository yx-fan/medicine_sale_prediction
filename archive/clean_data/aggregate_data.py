import pandas as pd

def aggregate_data(df):
    grouped_df = df.groupby(['药品名称', '厂家', 'start_date', 'end_date'], as_index=False).agg({
        '减少数量': 'sum',
        '增加数量': 'sum',
        '期初金额(进价)': 'sum'
    })
    
    grouped_df['减少数量总和'] = grouped_df.groupby(['药品名称', '厂家'])['减少数量'].transform('sum')
    result_df = grouped_df[grouped_df['减少数量总和'] != 0].copy()
    result_df.drop(columns=['减少数量总和'], inplace=True)
    return result_df