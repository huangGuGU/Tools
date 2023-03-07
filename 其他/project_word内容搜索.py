#########################################################################
# 目标：实现一个批量读取file里word，然后按照名字找到关键词写入excel
# File_path存的是所有word文档路径
# Excel_path是写入的excel路径
#########################################################################
from 装饰器.decorator_程序启动 import logit
import os
import docx
import pandas as pd


@logit
def find_sentence(path, excel_path):
    name_list = []
    key_word_list = []
    court_list = []
    data_list = []
    level_list = []
    word_list = os.listdir(path)
    try:
        word_list.remove('.DS_Store')
    except:
        pass

    for word in word_list:
        court = '无'
        data = '无'
        key_num = 1
        key_word = ''
        word_path = os.path.join(path, word)

        name_list.append(f'{word[:-5]}')
        if '二审' in word:
            level_list.append('二审')
        elif '一审' in word:
            level_list.append('一审')
        elif '三审' in word:
            level_list.append('三审')
        else:
            level_list.append('无')

        word = docx.Document(word_path)
        for para in word.paragraphs:

            if '医疗救助' in para.text:
                sentence = str(key_num) + '.' + para.text + '--------------------------------------------'
                key_word += sentence
                key_num += 1
            elif '审理法院' in para.text:
                court = para.text

            elif '裁判日期' in para.text:
                data = para.text
            else:
                pass
        key_word_list.append(key_word)
        court_list.append(court)
        data_list.append(data)

    df = pd.DataFrame(
        {'文件名': name_list, '法院': court_list, '审级': level_list, '时间': data_list, '句子': key_word_list})
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name='数据', float_format='%.6f')


if __name__ == '__main__':
    File_path = r'/Users/hzh/Desktop/file'
    Excel_path = r'/Users/hzh/Desktop/Word提取数据.xlsx'
    find_sentence(File_path, Excel_path)
