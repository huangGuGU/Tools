#########################################################################
# cut_num 代表的就是切割的个数，原图切割成cut_num*cut_num个
#########################################################################
import cv2
import os
from 装饰器.decorator_程序启动 import logit


@logit
def cut_patch(path, save, num):
    if not os.path.exists(save):
        os.makedirs(save)

    img = cv2.imread(path)
    h = img.shape[0]
    w = img.shape[1]

    small_h = h // num
    small_w = w // num
    num = 0
    for i in range(num):
        for j in range(num):
            x1 = i * small_h
            x2 = small_h + i * small_h
            y1 = j * small_w
            y2 = small_w + j * small_w
            image = img[x1:x2, y1:y2]

            target_path = os.path.join(save, f'img_{num}.png')
            cv2.imwrite(target_path, image)
            num += 1


if __name__ == '__main__':
    file_path = ''
    save_path = ''
    cut_num = 4
    cut_patch(file_path, save_path, cut_num)
