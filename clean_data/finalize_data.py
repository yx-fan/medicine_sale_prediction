import numpy as np

def finalize_data(df):
    df['total_期初金额_by_药品名称'] = df.groupby(['药品名称', 'start_date'])['期初金额(进价)'].transform('sum')
    df['期初金额占比'] = np.where(df['期初金额(进价)'] == 0, 0, df['期初金额(进价)'] / df['total_期初金额_by_药品名称'])
    
    df = df.sort_values(by=['药品名称', '厂家', 'start_date'])
    df['previous_增加数量'] = df.groupby(['药品名称', '厂家'])['增加数量'].shift(1).fillna(0)
    df.drop(columns=['total_期初金额_by_药品名称'], inplace=True)
    return df