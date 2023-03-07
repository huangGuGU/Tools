#
#
from 装饰器.decorator_程序启动 import logit
import cv2
import matplotlib.pyplot as plt
import numpy as np

@logit
def drwa_static(path,save_path):
    img = cv2.imread(path)
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    ret, thresh = cv2.threshold(G, 200, 255, cv2.THRESH_BINARY)
    Img = np.zeros((thresh.shape[0], thresh.shape[1], 3))
    for i in range(thresh.shape[0]):
        for j in range(thresh.shape[1]):
            if thresh[i,j]==255:
                Img[i, j, 0] = 255
                Img[i, j, 1] = 255
                Img[i, j, 2] = 255
            else:
                Img[i, j, 0] = 208
                Img[i, j, 1] = 117
                Img[i, j, 2] = 53

    cv2.imwrite(save_path, Img)


if __name__ == '__main__':
    path = r'/Users/hzh/Desktop/WechatIMG366.png'
    save_path = '/Users/hzh/Desktop/result.jpg'
    drwa_static(path,save_path)

