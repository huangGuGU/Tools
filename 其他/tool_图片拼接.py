#########################################################################
# img_file_path存放需要拼接的图片，如果想要按照指定规律拼接，可以用数字.png命名
# save_path是拼接过后的图片
# model 1是竖着拼，0是横着拼
#########################################################################
from 装饰器.decorator_程序启动 import logit
import os
import cv2
import numpy as np


@logit
def img_stitching(path, model, save):
    img_list = os.listdir(path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass
    img_container = []
    H_sum = 0
    W_sum = 0
    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img_container.append(img)
        H_sum += img.shape[0]
        W_sum += img.shape[1]

    if model == 1:
        W_Max = max(img_container, key=lambda x: x.shape[1]).shape[1]
        img_result = np.ones((H_sum, W_Max, 3)) * 255
        H = 0
        for index, I in enumerate(img_container):
            W_padding = (W_Max - I.shape[1]) // 2
            img_result[H:H + I.shape[0], W_padding:W_padding + I.shape[1], :] = I
            H = H + I.shape[0]
        cv2.imwrite(save, img_result)

    elif model == 0:
        H_Max = max(img_container, key=lambda x: x.shape[0]).shape[0]
        img_result = np.ones((H_Max, W_sum, 3)) * 255
        W = 0
        for index, I in enumerate(img_container):
            H_padding = (H_Max - I.shape[0]) // 2
            img_result[H_padding:H_padding + I.shape[0], W:W + I.shape[1], :] = I
            W = W + I.shape[1]
        cv2.imwrite(save, img_result)

    else:
        print('模式输入错误，输入0/1')


if __name__ == '__main__':
    stitch_model = 1
    img_file_path = r'/Users/hzh/Desktop/img'
    save_path = rf'/Users/hzh/Desktop/img_stitched_{stitch_model}.png'
    img_stitching(img_file_path, stitch_model, save_path)
