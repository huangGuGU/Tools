import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# path = '/Users/hzh/手写数据集/test_img/7'
# target = '/Users/hzh/Desktop/test_img_choice/7'
# img_list = os.listdir(path)
# if not os.path.exists(target):
#     os.makedirs(target)
#
# try:
#     img_list.remove('.DS_Store')
# except:
#     pass
# num = []
# m = 1
# for img in img_list:
#     total_middle = 0
#     img_path = os.path.join(path, img)
#     target_path = os.path.join(target, img)
#     data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     zero_num = np.sum(np.where(data, 0, 1))
#
#     zero_ratio = zero_num / (data.shape[0] * data.shape[1])
#     num.append(zero_ratio)
#     if zero_ratio < 0.84 and zero_ratio<0.85:
#         cv2.imwrite(target_path, data)
#
# n, bins, patches = plt.hist(np.array(num), 20)
# plt.show()
# print('done')





number = '5'
clazz = 'test'
path = f'/Users/hzh/手写数据集/{clazz}_img/{number}'
# path = f'/Users/hzh/Desktop/{number}'
# path = f'/Users/hzh/手写数据集/{clazz}_choise/{number}'
target = f'/Users/hzh/Desktop/{clazz}_imgs_choice/{number}'
img_list = os.listdir(path)
if not os.path.exists(target):
    os.makedirs(target)
try:
    img_list.remove('.DS_Store')
except:
    pass
num = []
L1_length = []
L2_length = []
m = 1
for img in img_list:
    total_middle = 0
    img_path = os.path.join(path, img)
    target_path = os.path.join(target, img)
    data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    data_1 = cv2.imread(f'/Users/hzh/手写数据集/{clazz}_img/{number}/{img}', cv2.IMREAD_GRAYSCALE)
    zero_num = np.sum(np.where(data, 0, 1))

    zero_ratio = zero_num / (data.shape[0] * data.shape[1])
    num.append(zero_ratio)
    v1 = v2 = v3 = v4 = 0
    a1 = a2 = b1 = b2 = 0
    m, n = data.shape
    for i in range(0, m, 1):
        for j in range(0, n, 1):
            if data[i, j] >= 1:
                b1 = i
                v1 = 1
                break
        if v1 == 1:
            break
    for i in range(0, n, 1):
        for j in range(0, m, 1):
            if data[j, i] >= 1:
                a1 = i
                v2 = 1
                break
        if v2 == 1:
            break
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if data[i, j] >= 1:
                b2 = i
                v3 = 1
                break
        if v3 == 1:
            break
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if data[j, i] >= 1:
                a2 = i
                v4 = 1
                break
        if v4 == 1:
            break
    L1 = a2 - a1  # 左右的长度
    L2 = b2 - b1  # 上下的长度
    L1_length.append(L1)
    L2_length.append(L2)

    # if ((zero_ratio >= 0.76) and (zero_ratio <= 0.82))and\
    #    ((L1 >= 14) and (L1 <= 17))and\
    #    ((L2 >= 17) and (L2 <= 19)):
    #     cv2.imwrite(target_path, data_1)
plt.figure()
n, bins, patches = plt.hist(np.array(num),20)
plt.title('ratio')
plt.figure()
n1, bins1, patches1 = plt.hist(np.array(L1_length))
plt.title('width')
plt.figure()
n2, bins2, patches2 = plt.hist(np.array(L2_length))
plt.title('height')
plt.show()
print('done')