#########################################################################
# kernel_x ,kernel_y 是卷积核的大小，核越大，出来的图片数量越少
# step是核之间的间距像素
#########################################################################
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from 装饰器.decorator_程序启动 import logit


@logit
def cut_img_by_kernel():
    img_list = os.listdir(path)
    if not os.path.exists(save):
        os.makedirs(save)
    try:
        img_list.remove('.DS_Store')
    except:
        pass

    for img in img_list:

        img_path = os.path.join(path, img)
        Img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        length = Img.shape[0]
        height = Img.shape[1]

        n = int((length - kernel_x) / step) + 1
        m = int((height - kernel_y) / step) + 1
        print("图片总数", n * m)
        input('是否继续')
        num = 0
        for k1 in range(n):
            for k2 in range(m):
                x1 = k1 * step
                x2 = kernel_x + k1 * step
                y1 = k2 * step
                y2 = kernel_y + k2 * step
                image = Img[x1:x2, y1:y2]
                target_path = os.path.join(save, f'{img[:-4]}_{num}.png')
                cv2.imwrite(target_path, image)
                num += 1


if __name__ == '__main__':
    path = r'../images/img'
    save = r'../images/save'
    kernel_x = 80
    kernel_y = 80
    step = 65
    cut_img_by_kernel()
