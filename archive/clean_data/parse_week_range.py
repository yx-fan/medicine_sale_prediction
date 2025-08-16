from datetime import datetime

def parse_week_range(week_range, file_year):
    start_str, end_str = week_range.split('-')
    start_month_day = start_str.split('.')
    if '.' in end_str:
        end_month_day = end_str.split('.')
    else:
        end_month_day = [start_month_day[0], end_str]
    
    start_month, start_day = int(start_month_day[0]), int(start_month_day[1])
    end_month, end_day = int(end_month_day[0]), int(end_month_day[1])

    start_year = int(file_year)
    end_year = start_year + 1 if start_month > end_month else start_year

    start_date = datetime(year=start_year, month=start_month, day=start_day)
    end_date = datetime(year=end_year, month=end_month, day=end_day)
    
    return start_date, end_date
