# 获取某年某月是周几
from datetime import datetime
import numpy as np
import pandas as pd
import calendar

year = int("20"+input('年份：'))
month = int(input('月份：'))
path = r'/Users/hzh/Desktop/today.xlsx'

name_list = ["周几", '嵇友鹏', '吴明珠', '成意', '黄桂锋', "龙向东"]
maxday = int(calendar.monthrange(2022, 11)[1])  # 当月的天数
today_array = np.zeros((len(name_list), maxday))

for day in range(1, maxday + 1):

    week = datetime.strptime(f"{year}{month}{day}", "%Y%m%d").weekday() + 1
    today_array[0, day - 1] = week
    if week in [1, 2, 3, 4, 5]:
        for a in [2, 3]:
            today_array[a, day - 1] = 1
            today_array[a, day - 1] = 1
    else:
        for a in [2, 3]:
            today_array[1, day - 1] = 0
            today_array[3, day - 1] = 0

    for b in [1, 4]:
        today_array[b, :] = 0
        today_array[b, :] = 0

dataFrame = pd.DataFrame(today_array, index=name_list, columns=np.arange(1, maxday + 1))

with pd.ExcelWriter(path) as writer:
    dataFrame.to_excel(writer, sheet_name=f'{year} {month}')
print('done')