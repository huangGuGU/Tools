import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import os

path = '/Users/hzh/手写数据集/test_img/7'
target = '/Users/hzh/Desktop/test_img_choice/7'
img_list = os.listdir(path)
if not os.path.exists(target):
    os.makedirs(target)
try:
    img_list.remove('.DS_Store')
except:
    pass
num = []
m = 1
for img in img_list:
    total_middle = 0
    img_path = os.path.join(path, img)
    target_path = os.path.join(target, img)
    data = cv2.imread(img_path)
    data_k = np.fft.fft2(data)
    data_k = np.fft.fftshift(data_k)
    data_k = np.abs(data_k)
    data_k = np.log(data_k + 1)
    data_k = np.log(data_k.flatten() + 1)

    # n, bins, patches = plt.hist(data_k,20)

    for i in range(len(data_k)):
        if data_k[i] > 0.5 * (max(data_k) - min(data_k)):
            total_middle += data_k[i]

    num.append(total_middle)
    if total_middle > 1544 and total_middle<1549:
        cv2.imwrite(target_path, data)

n, bins, patches = plt.hist(np.array(num))
plt.show()
print('done')