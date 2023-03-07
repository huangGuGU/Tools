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
def img_to_frequency(path,save):
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
        im = cv2.imread(img_path, -1)
        im = cv2.resize(im, (240, 240))

        im = np.array(im)
        plt.subplot(141)
        plt.title('image')
        plt.axis('off')
        plt.imshow(im)
        plt.colorbar()


        im_k = np.fft.fft2(im)
        im_k_shift = np.fft.fftshift(im_k)
        im_k_shift_abs = np.abs(im_k_shift)
        plt.subplot(142)
        plt.title('k_shift')
        plt.axis('off')
        plt.imshow(im_k_shift_abs)
        plt.colorbar()

        im_k_log = np.log(im_k_shift_abs)
        plt.subplot(143)
        plt.title('k_log')
        plt.axis('off')
        plt.imshow(im_k_log)
        plt.colorbar()


        img_fft = np.fft.ifft2(im_k_shift)
        img_fft_abs = np.abs(img_fft)
        plt.subplot(144)
        plt.title('second fft')
        plt.axis('off')
        plt.imshow(img_fft_abs)
        plt.colorbar()


        plt.tight_layout()
        plt.show()

        out_img(im_k, save_path)
        print(f'{name} done')


if __name__ == '__main__':
    path = r'/Users/hzh/Desktop/img'
    save = r'/Users/hzh/Desktop/img_k'
    img_to_frequency(path,save)
    pass
