#####################################################################################
# TXT_path存放1k个顺序中文名
# Excel_path存放分类好的excel文件
#####################################################################################
import os
import pandas as pd
import PIL.Image as Image

'''获取excel的每一个类中内容对应ImagNet的图片文件夹位置'''


def get_index(excel_path, txt_path):
    class_index = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        n_list = f.readlines()
        name_list = list(map(lambda x: x[:-1], n_list))  # 去掉\n

    df = pd.read_excel(excel_path, sheet_name='Sheet1')
    for Line in range(df.shape[1]):
        class_line = df.iloc[:, Line].dropna(axis=0, how='all')
        class_line = class_line.tolist()
        clazz = df.columns[Line]

        name_index = list(map(lambda x: name_list.index(x), class_line))  # 获取这一类的物品在imagenet的图片文件夹位置
        class_index[clazz] = name_index
    return class_index


def save_img(img_path, class_index, save_path):
    img_list = os.listdir(img_path)

    for clazz, index in class_index.items():
        class_path = os.path.join(save_path, clazz)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

        for i in index:

            file_name = img_list[i]

            image_path = os.path.join(img_path, file_name)
            imgs = os.listdir(image_path)

            for img in imgs:
                im_path = os.path.join(image_path, img)
                im = Image.open(im_path)
                im_save_path = os.path.join(save_path, clazz, img)
                im.save(im_save_path)
        print(f'{clazz} done')

    print('all done')


if __name__ == '__main__':
    Excel_path = r'分类.xlsx'
    TXT_path = r'imagenet类别.txt'
    Img_path = r'E:\download\train_choice 400 595'
    Save_path = r'E:\download\big_class_choice'

    Class_index = get_index(Excel_path, TXT_path)
    save_img(Img_path, Class_index, Save_path)
