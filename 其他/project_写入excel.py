from 装饰器.decorator_程序启动 import logit
import os

import numpy as np
# import pandas as pd
#
#
# path = r'/Users/hzh/Desktop/x.xlsx'
# df = pd.DataFrame({'Name': ['Alice', 'Bob\nss\nss\nss', 'Charlie\n'], 'Age': [25, 30, 35]})
#
# with pd.ExcelWriter(path) as writer:
#     df.to_excel(writer,sheet_name='onn级联unet',float_format='%.6f')
# print('done')

key_word = ''
for pra in ['1','2','3']:
    sentence = pra+'.' + '----------------------------------------------------------------------------------------'
    key_word += sentence
print(key_word)