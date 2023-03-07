import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/hzh/Desktop/x.png'
save = '/Users/hzh/Desktop/c.png'
save_edge = '/Users/hzh/Desktop/edge.png'
save_colour = '/Users/hzh/Desktop/colour.png'
img = cv2.imread(path, 0)


def preProccessing(img):
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    imgDial = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=2)  # 膨胀操作
    imgThres = cv2.erode(imgDial, np.ones((5, 5)), iterations=1)  # 腐蚀操作
    return imgThres


img = preProccessing(img)
cv2.imwrite(save, img)

'''find edge'''
hight, length = img.shape
h_list = []
l_list = []
for H in range(hight // 2, hight):
    for L in range(length):
        if img[H, L] == 255:
            h_list.append(H)
            l_list.append(L)
            break

'''confirm edge'''
def confirm_edge(img):
    IMG_OUT = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for m in range(len(h_list)):
        IMG_OUT[h_list[m], l_list[m], 0] = 255
        IMG_OUT[h_list[m], l_list[m], 1] = 0
        IMG_OUT[h_list[m], l_list[m], 2] = 0
    cv2.imwrite(save_edge, IMG_OUT)


confirm_edge(img)

'''find point'''
x = l_list[:160]
y = h_list[:160]  # 画图时候原点在左下，但是np索引是在左上,如果plot需要反转

'''linear fit'''
from sklearn.linear_model import LinearRegression

x = np.array(x)
y = np.array(y)
x = np.expand_dims(x, 1)
y = np.expand_dims(y, 1)
lin_reg = LinearRegression()
lin_reg.fit(x, y)
x_new = np.linspace(48, 160, 200)
x_new = np.expand_dims(x_new, 1)
y_new = lin_reg.predict(x_new).astype(np.int)
x_new = x_new.astype(np.int)

'''plot on original'''
img_rgb = cv2.imread(path)
for pixel in range(200):
    a = x_new[pixel]
    b = y_new[pixel]
    img_rgb[y_new[pixel], x_new[pixel], 0] = 255
    img_rgb[y_new[pixel], x_new[pixel], 1] = 0
    img_rgb[y_new[pixel], x_new[pixel], 2] = 0
cv2.imwrite(save_colour, img_rgb)
plt.show()
