#########################################################################
# path 是目标图片文件夹路径
# save 是保存增强后图片的文件夹路径
# show_flag 是否展示单个图片的所有的增强图片
# 不需要使用的增强方法，只要将img_aug_tuple内容注释即可
#########################################################################
from 装饰器.decorator_程序启动 import logit
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations import (RandomRotate90, ShiftScaleRotate, HorizontalFlip, Transpose, VerticalFlip, Rotate)


def ImageRotate(imagepath):
    image = cv2.imread(imagepath)
    # 要有中心坐标、旋转角度、缩放系数
    h, w = image.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    center = (w // 2, h // 2)  # 绕图片中心进行旋转
    angle = 45  # 旋转角度
    scale = 0.8  # 将图像缩放为80%

    # 1. 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=scale)  # 当angle为负值时，则表示为顺时针

    # 2. 进行仿射变换，borderValue:缺失背景填充色彩，默认是黑色（0, 0 , 0），这里指定填充白色
    # 注意，这里的dsize=(w, h)顺序不要搞反了
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(w, h), borderValue=(255, 255, 255))

    return image_rotation


@logit
def img_albumentations(path, save, show_flag):
    img_list = os.listdir(path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass
    if not os.path.exists(save):
        os.makedirs(save)

    for img in img_list:
        img_name = img[:-4]
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path)
        # img[-200:-1, 0:200, :] = 0  # mask 左下角单词处理
        img = img.astype(np.uint8)
        img1 = np.rot90(img)
        img2 = np.rot90(img1)
        img3 = np.rot90(img2)

        # 多个处理

        # img_transpose = Transpose(p=1)(image=img)['image']

        img_horizontal = HorizontalFlip(p=1)(image=img)['image']
        img_horizontal1 = np.rot90(img_horizontal)
        img_horizontal2 = np.rot90(img_horizontal1)
        img_horizontal3 = np.rot90(img_horizontal2)

        img_vertical = VerticalFlip(p=1)(image=img)['image']
        img_vertical1 = np.rot90(img_vertical)
        img_vertical2 = np.rot90(img_vertical1)
        img_vertical3 = np.rot90(img_vertical2)

        img_shift_scale_rotate = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=1) \
            (image=img)['image']

        img_shift_scale_rotate1 = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=1) \
            (image=img)['image']

        img_shift_scale_rotate2 = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=1) \
            (image=img)['image']

        img_shift_scale_rotate3 = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=1) \
            (image=img)['image']

        img_shift_scale_rotate4 = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1) \
            (image=img)['image']

        img_aug_tuple = (
            img,
            img1,
            img2,
            img3,

            # img_transpose,
            img_vertical,
            img_vertical1,
            img_vertical2,
            img_vertical3,

            # img_shift_scale_rotate,
            # img_shift_scale_rotate1,
            # img_shift_scale_rotate2,
            # img_shift_scale_rotate3,
            # img_shift_scale_rotate4
        )

        def img_all_show(img_aug_tuple):
            for index, img_aug in enumerate(img_aug_tuple):
                img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
                plt.subplot(4, 4, index + 1)
                plt.imshow(img_aug)
                plt.title(index + 1)
                plt.axis('off')
            plt.show()

        if show_flag:
            img_all_show(img_aug_tuple)
            input('按任意键展示下一张')

        # 保存文件
        for index, img_aug in enumerate(img_aug_tuple):
            save_path = os.path.join(save, img_name + '_' + str(index) + '.png')
            cv2.imwrite(save_path, img_aug)
        print(f'{img_name} done')


if __name__ == '__main__':
    path = r'/Users/hzh/Desktop/九院肠息肉'
    save = r'/Users/hzh/Desktop/label'
    show_flag = False
    img_albumentations(path, save, show_flag)
