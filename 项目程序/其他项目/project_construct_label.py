#########################################################################
#
# 按照2：8的分割将image里的图片都分成train和test
#
#########################################################################
import os
import random

path = r'image'
all_sample_txt = 'label.txt'
train_sample_txt = 'label_train.txt'
test_sample_txt = 'label_test.txt'
all_random = True
k = 0.8

sample_list = []
file_list = os.listdir(path)

for file in file_list:
    file_path = os.path.join(path,file)
    img_list = os.listdir(file_path)
    for image in img_list:
        # clazz = image.split('_')[2][:-4]
        img_path = os.path.join(file_path,image)
        sample_list.append(f'{img_path} {file}')


with open(all_sample_txt, 'w') as f:
    for i in sample_list:
        f.writelines(i)
        f.writelines('\n')

index_list = list(range(len(sample_list)))

random.shuffle(index_list)

train_len = int(len(sample_list)*k)


with open(train_sample_txt, 'w') as f1:
    for n in range(train_len):
        f1.writelines(sample_list[index_list[n]])
        f1.writelines('\n')

with open(test_sample_txt, 'w') as f2:
    for n in range(train_len,len(sample_list)):
        f2.writelines(sample_list[index_list[n]])
        f2.writelines('\n')
    pass
