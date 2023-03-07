#########################################################################
#
# test和train的图片已经分好后，通过两个文件夹创建索引
#
#########################################################################
import os
import random

path_test = r'img_test'
path_train = r'img_train'

train_sample_txt = 'label_train.txt'
test_sample_txt = 'label_test.txt'


def construct_label(path,txt):
    sample_list = []
    file_list = os.listdir(path)

    for file in file_list:
        file_path = os.path.join(path,file)
        img_list = os.listdir(file_path)
        for image in img_list:
            name = image.split('_')
            x = name[1]
            y = name[2][:-4]
            img_path = os.path.join(file_path,image)
            sample_list.append(f'{img_path} {x}_{y}')

    index_list = list(range(len(sample_list)))
    random.shuffle(index_list)
    Len = int(len(sample_list))

    with open(txt, 'w') as f1:
        for n in range(Len):
            f1.writelines(sample_list[index_list[n]])
            f1.writelines('\n')


for i in [[path_test,test_sample_txt],[path_train,train_sample_txt]]:
    construct_label(i[0],i[1])
