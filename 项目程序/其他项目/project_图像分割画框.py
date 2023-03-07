#########################################################################
# path是图片路径，m和M分别是需要抠的图的最大和最小值，设置好后可以避免不必要的抠图
# 实现：抠出的图片plt出来查看
#########################################################################
from matplotlib import pyplot as plt
from 装饰器.decorator_程序启动 import logit
import cv2


def Process(imgCopy):
    imgCopy = 255 - imgCopy
    ret, img = cv2.threshold(imgCopy, 0, 255, cv2.THRESH_OTSU)
    return img


@logit
def draw_square(path, m, M):
    img_list = []

    img = cv2.imread(path, 0)
    imgcopy = img.copy()
    img_process = Process(imgcopy)

    contours, hierarchy = cv2.findContours(img_process, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > m and area < M:  # 面积大于像素记为封闭图形
            peri = cv2.arcLength(cnt, True)  # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 计算有多少个拐角
            x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小
            img_get = img[y:y + h, x:x + w]
            img_list.append(img_get)

    for img in img_list:
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    path = r'/Users/hzh/Desktop/WechatIMG513.jpeg'
    draw_square(path, m=0, M=1000000)
