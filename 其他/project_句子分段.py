import os

path = '/Users/hzh/Desktop/xx.txt'
target = '/Users/hzh/Desktop/yy.txt'
txt = ''
with open(path, 'r') as f:
    lines = f.readlines()  # 读取数据
    for Line in lines:
        txt += Line

Line = txt.split('.')[:-1]

for index, sentence in enumerate(Line):
    # process_line = f'第{index + 1}句:  {sentence}.'
    # with open(target, 'a') as f2:
    #     f2.writelines(process_line)
    #     f2.writelines('\n')
    print(f'第{index + 1}句:  {sentence}.')
# print('done')
