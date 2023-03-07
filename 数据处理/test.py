import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
#
#
#
# path = r'/Users/hzh/Desktop/psfimage 2'
# label = r'/Users/hzh/Desktop/8156_90.0_55.0.png'
# save = r'/Users/hzh/Desktop/save'
# img_list = os.listdir(path)
# for i in img_list:
#     I = i.split('_')
#     name = f'{I[1]}_{I[2]}'
#     img_path = os.path.join(path,i)
#     save_path = os.path.join(save,name)
#
#     img = cv2.imread(img_path)
#
#     cv2.imwrite(save_path, img)
#
# num = len(os.listdir(save))
# print(num)

# detector = pd.read_csv("/Users/hzh/Desktop/detectorlines56.csv")
# print(detector.shape)
# d = np.array(detector)
# d_list = []
# for i in range(10):
#     D = d[i,:]
#     D = D.reshape(56,56)
#     plt.imshow(D)
#
#     plt.savefig(rf'/Users/hzh/Desktop/{i}.png')
#     plt.show()


