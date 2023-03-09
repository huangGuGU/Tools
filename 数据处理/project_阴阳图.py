#########################################################################
# File_path是输出图片文件夹路径，Label_path是label路径，文件里图片名字需要相同
# Save_path保存阴阳图的路径
# 假阴 紫色；假阳 绿色
#########################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from 装饰器.decorator_程序启动 import logit


@logit
def fp_img(path, label_path, save_path):
    img_list = os.listdir(path)
    label_list = os.listdir(label_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    try:
        img_list.remove('.DS_Store')
        label_list.remove('.DS_Store')
    except:
        pass

    '''判断阴阳'''
    if len(img_list) > 0:
        for img in img_list:

            img_path = os.path.join(path, img)
            Label_path = os.path.join(label_path, img)
            Save_path = os.path.join(save_path, img)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(Label_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, img.shape)

            a, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
            # plt.imshow(img)
            # plt.show()

            b, label = cv2.threshold(label, 0, 255, cv2.THRESH_OTSU)
            # plt.imshow(label)
            # plt.show()

            size0 = img.shape[0]
            size1 = img.shape[1]
            Img = np.zeros((size0, size1, 3))

            for i in range(size0):
                for j in range(size1):

                    if img[i, j] == 0 and label[i, j] == 0:  # 真阴
                        Img[i, j, :] = 0
                    elif img[i, j] == 0 and label[i, j] == 255:  # 假阴 紫色
                        Img[i, j, 0] = 0.4940 * 255
                        Img[i, j, 1] = 0.1840 * 255
                        Img[i, j, 2] = 0.5560 * 255
                    elif img[i, j] == 255 and label[i, j] == 0:  # 假阳 绿色
                        Img[i, j, 0] = 0.1880 * 255
                        Img[i, j, 1] = 0.6740 * 255
                        Img[i, j, 2] = 0.4660 * 255
                    else:  # 真阳
                        Img[i, j, :] = 255
            cv2.imwrite(Save_path, Img)


if __name__ == '__main__':
    Label_path = '../images/label'
    File_path = r'../images/img'
    Save_path = r'../images/save'
    fp_img(File_path, Label_path, Save_path)
