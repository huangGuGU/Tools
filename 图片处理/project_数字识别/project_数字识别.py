from 图片处理.project_数字识别.model import Hao
import torch
import numpy as np
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt


def preProccessing(img):
    imgGray = 255-cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    plt.imshow(imgBlur, 'gray')
    plt.axis('off')
    # plt.show()
    # imgcanny = cv2.Canny(imgGray, 11,11)
    # plt.imshow(imgcanny , 'gray')
    # plt.axis('off')
    # plt.show()

    imgThres = cv2.erode(imgBlur, np.ones((5, 5)), iterations=1)  # 腐蚀操作
    plt.imshow(imgThres,'gray')
    plt.axis('off')
    # plt.show()
    ret, thresh = cv2.threshold(imgThres, 0, 255, cv2.THRESH_OTSU)


    imgDial = cv2.dilate(thresh  , np.ones((5,5)), iterations=2)  # 膨胀操作
    plt.imshow(imgDial,'gray')
    plt.axis('off')
    # plt.show()





    return imgDial



def getContours(img):
    x, y, w, h, xx, yy, ss = 0, 0, 10, 10, 20, 20, 10  # 因为图像大小不能为0
    imgGet = np.array([[], []])  # 不能为空
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # 面积大于800像素为封闭图形
            cv2.drawContours(imgCopy, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)  # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 计算有多少个拐角
            x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小
            a = (w + h) // 2
            dd = abs((w - h) // 2)  # 边框的差值
            imgGet = imgProcess[y:y + h, x:x + w]
            if w <= h:  # 得到一个正方形框，边界往外扩充20像素,黑色边框
                imgGet = cv2.copyMakeBorder(imgGet, 20, 20, 20 + dd, 20 + dd, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                xx = x - dd - 10
                yy = y - 10
                ss = h + 20
                # cv2.rectangle(imgCopy, (x - dd - 10, y - 10), (x + a + 10, y + h + 10), (0, 255, 0),
                #               2)  # 看看框选的效果，在imgCopy中
                # print(a + dd, h)
            else:  # 边界往外扩充20像素值
                imgGet = cv2.copyMakeBorder(imgGet, 20, 20, 20 + dd, 20 + dd, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                xx = x - 10
                yy = y - dd - 10
                ss = w + 20
                # cv2.rectangle(imgCopy, (x - 10, y - dd - 10), (x + w + 10, y + a + 10), (0, 255, 0), 2)
                # print(a + dd, w)
            Temptuple = (imgGet, xx, yy, ss)  # 将图像及其坐标放在一个元组里面，然后再放进一个列表里面就可以访问了
            Borderlist.append(Temptuple)

    return Borderlist



Borderlist = []  # 不同的轮廓图像及坐标
Resultlist = []  # 识别结果
img = cv2.imread('./num/2.png')

imgCopy = img.copy()
imgProcess = preProccessing(img)
Borderlist = getContours(imgProcess)




train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])




model = Hao()
model = torch.load('./Hao.pth')


model.eval()
index_to_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Result = {}

if len(Borderlist) != 0:  # 不能为空
    for (imgRes, x, y, s) in Borderlist:
        # cv2.imshow('imgCopy', imgRes)
        # cv2.waitKey(0)
        img = train_transform(imgRes)
        plt.imshow(img[0,:,:])
        plt.show()
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = model(img)
            output = torch.squeeze(output)



            predict_cla = torch.argmax(output).numpy()
            # print(index_to_class[predict_cla], output[predict_cla].numpy())
            result = index_to_class[predict_cla]
            Result[x] = result





        # cv2.rectangle(imgCopy, (x, y), (x + s, y + s), color=(0, 255, 0), thickness=10)
        cv2.putText(imgCopy, result, (x + s // 2 - 5,  y + s // 2-110 ), cv2.FONT_HERSHEY_COMPLEX, 1.5, (10,0,255), 2)
plt.figure(dpi=300)
plt.imshow( imgCopy)
plt.axis('off')
plt.show()



