#########################################################################
# color_flag=false代表图片是灰度图，否则就是RGB
#########################################################################
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from 装饰器.decorator_程序启动 import logit


@logit
def find_diff(path, target, save, flag):
    if not os.path.exists(save):
        os.mkdir(save)
    img_list = os.listdir(path)
    label_list = os.listdir(target)
    try:
        img_list.remove('.DS_Store')
        label_list.remove('.DS_Store')
    except:
        pass
    for img in img_list:
        img_path = os.path.join(path, img)
        target_path = os.path.join(target, img)
        save_path = os.path.join(save, img)

        if flag:
            im1 = cv.imread(img_path)
            im2 = cv.imread(target_path)
            im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)
            im2 = cv.cvtColor(im2, cv.COLOR_BGR2RGB)
            im1 = np.float32(im1 / 255)
            im2 = np.float32(im2 / 255)
            Different_point = np.where(im1 != im2)
            im1[Different_point[0][:], Different_point[1][:], 2] = 0
            im1[Different_point[0][:], Different_point[1][:], 1] = 255
            im1[Different_point[0][:], Different_point[1][:], 0] = 0
            plt.imshow(im1)
            plt.axis('off')
            plt.show()

        else:
            Im1 = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            Im2 = cv.imread(target_path, cv.IMREAD_GRAYSCALE)
            Im3 = np.abs(Im1 - Im2)
            print(Im3.max())

        plt.imshow(Im3)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    file_path = '/Users/hzh/Desktop/output'
    label_path = '/Users/hzh/Desktop/label'
    Save_path = '/Users/hzh/Desktop/result'
    color_flag = False
    find_diff(file_path, label_path, Save_path, color_flag)
