# 
# 
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from 装饰器.decorator_程序启动 import logit


def out_img(img_out, target_path):
    plt.rcParams['figure.figsize'] = (10.24, 10.24)
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.imshow(img_out, cmap='gray')
    plt.savefig(target_path)


@logit
def frequency_to_space(path,save):
    if not os.path.exists(save):
        os.mkdir(save)
    img_list = os.listdir(path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass

    for name in img_list:
        img_path = os.path.join(path, name)
        save_path = os.path.join(save, name)
        im_k = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        im_k = cv2.resize(im_k, (240, 240))

        im = np.fft.ifft2(im_k)
        im = np.abs(im)

        out_img(im, save_path)
        print(f'{name} done')


if __name__ == '__main__':
    path = r'/Users/hzh/Desktop/img_k'
    save = r'/Users/hzh/Desktop/img_k_to_space'
    frequency_to_space(path,save)
    pass
