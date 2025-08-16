import numpy as np

def finalize_data(df):
    # 计算每个药品在每个日期的期初金额总和
    df['total_期初金额_by_药品名称'] = df.groupby(['药品名称', 'start_date'])['期初金额(进价)'].transform('sum')

    # 初始化期初金额占比
    df['期初金额占比'] = np.where(df['期初金额(进价)'] == 0, 0, df['期初金额(进价)'] / df['total_期初金额_by_药品名称'])

    # 检查每个药品是否只有一个厂家，如果是且期初金额为0，则将占比设为1
    single_factory = df.groupby('药品名称')['厂家'].transform('nunique') == 1
    zero_amount = df['期初金额(进价)'] == 0
    df.loc[single_factory & zero_amount, '期初金额占比'] = 1

    # 按药品名称、厂家和日期排序
    df = df.sort_values(by=['药品名称', '厂家', 'start_date'])

    # 添加前一期的增加数量
    df['previous_增加数量'] = df.groupby(['药品名称', '厂家'])['增加数量'].shift(1).fillna(0)

    # 删除不再需要的列
    df.drop(columns=['total_期初金额_by_药品名称'], inplace=True)
    
    return df
