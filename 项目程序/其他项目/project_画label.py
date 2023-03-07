# 
# 
from 装饰器.decorator_程序启动 import logit
import os
import cv2
import numpy as np



@logit
def draw_label(path, label, save):
    img_list = os.listdir(path)
    for i in img_list:
        img_path = os.path.join(path, i)
        label_path = os.path.join(label, i)
        save_path = os.path.join(save, i)
        img = cv2.imread(img_path)
        Label = cv2.imread(label_path, -1)

        index = np.where(Label == 0)
        for ii, jj in zip(index[0], index[1]):
            img[ii, jj] = 255
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    Path = r'/Users/hzh/Desktop/input'
    Label = r'/Users/hzh/Desktop/label'
    Save = r'/Users/hzh/Desktop'

    draw_label(Path, Label, Save)
