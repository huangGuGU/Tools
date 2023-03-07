"""如果是一个或多个中文出现，统一变成..."""

import re

path = '/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/Python/学习程序/知识点/xx.txt'
TxT = ''
with open(path) as f1:
    txt = f1.readlines()
    for line in txt:
        TxT = TxT + line
# xxxx
result = re.sub('[\u4e00-\u9fa5]+', '...', TxT)
print(result)
