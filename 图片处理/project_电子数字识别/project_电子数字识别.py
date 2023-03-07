import cv2
import numpy as np
from imutils import contours
def cv_show(img,img_name):
    cv2.imshow(img_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#第一步：模板处理
#读入模板图片
img = cv2.imread('./num.png')
#转成灰度图
img_gray = cv2.imread('./num.png',0)

#阈值处理
thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
#找出轮廓
temp_contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
#画出轮廓
img_copy = img.copy()
cv2.drawContours(img_copy,temp_contours,-1,(0,255,0),2)
# cv_show(img_copy,'temp_contours')
#轮廓排序
temp_contours = contours.sort_contours(temp_contours,method='left-to-right')[0]
#将数字与轮廓存入字典
temp_dict = {}
for i,contour in enumerate(temp_contours):
    x,y,w,h = cv2.boundingRect(contour)
    contour_img = thresh[y:y+h,x:x+w]
    contour_img = cv2.resize(contour_img,(55,88))
    temp_dict[i]=contour_img   # 把图片对应的数字存入字典







#第二步：卡片处理
kernel1 = np.ones([3,9])
kernel2= np.ones([6,6])
#读入卡片
card = cv2.imread('./card.png')
card= cv2.resize(card,(300,200))
#转成灰度图
card_gray = cv2.cvtColor(card,cv2.COLOR_BGR2GRAY)

#顶帽操作,突出
tophat = cv2.morphologyEx(card_gray,cv2.MORPH_TOPHAT,kernel1)
# cv_show(tophat,'tophat')
#边缘检测
sobel = cv2.Sobel(tophat,cv2.CV_64F,1,0,ksize=-1)
# cv_show(sobel,'sobel')
#取绝对值
sobel = np.absolute(sobel)
#归一化
(minVal,maxVal)=(np.min(sobel),np.max(sobel))
gradx=(255*((sobel-minVal)/(maxVal-minVal)))
gradx=gradx.astype("uint8")
#闭操作,填充小空隙
close = cv2.morphologyEx(gradx,cv2.MORPH_CLOSE,kernel1)
# cv_show(close,'close')
#阈值操作
card_thresh = cv2.threshold(close,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
# cv_show(card_thresh,'card_thresh')
#闭操作
thresh = cv2.morphologyEx(card_thresh,cv2.MORPH_CLOSE,kernel2)
# cv_show(close,'close')
#找到轮廓
card_copy = card.copy()
card_contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(card_copy,card_contours,-1,(0,0,255),2)
# cv_show(card_copy,'card_copy')




#第三步：找出卡片上要匹配的位置
#找到合适的区域locs
locs = []
for (i,contour) in enumerate(card_contours):
    x,y,w,h = cv2.boundingRect(contour)
    arc = w/float(h)
    if arc>2.5 and arc<4.5:
        if (w>40 and w<55) and (h>10 and h<20):
            locs.append((x,y,w,h))
locs.sort(key=lambda x:x[0])
print(np.shape(locs))
#遍历每个locs，找到每个数字的轮廓
predict_nums = []
predict_locs = []
for i,loc in enumerate(locs):
    x,y,w,h = loc
    card1 = card_gray[y-5:y+h+5,x-5:x+w+5]
    card1_thresh = cv2.threshold(card1,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    #进行轮廓检测，并对轮廓进行排序操作
    contours3,hirerchy = cv2.findContours(card1_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #对轮廓检测的结果按照从左到右的顺序排序
    contourses = contours.sort_contours(contours3,method="left-to-right")[0]

    #第四步：遍历轮廓，使用轮廓的外接矩阵获得数字图片，并使用cv2.resize改变图片大小
    for i,contour in enumerate(contourses):
        scores = []
        x1,y1,w1,h1 = cv2.boundingRect(contour)
        predict_locs.append((x1-6+x,y1-6+y,w1+2,h1+2))
        # print(predict_loc[:])
        contour_img = card1_thresh[y1:y1+h1,x1:x1+w1]
        contour_img = cv2.resize(contour_img,(55,88))
        # cv_show(contour_img,'contour_img')
        #进行模板匹配
        #遍历数字模板，使用matchTemplate找出与图片匹配度最高的数字
        for templates in temp_dict.values():
            ret = cv2.matchTemplate(contour_img, templates, cv2.TM_CCORR_NORMED)
            _, score, _, _ = cv2.minMaxLoc(ret)
            scores.append(score)
        predict_nums.append(str(np.argmax(scores)))
#第五步：在卡片上标出识别的数字
for i in range(len(predict_nums)):
    x,y,w,h = predict_locs[i]
    cv2.rectangle(card,(x-2,y-2),(x+w+2,y+h+2),(0,255,0),2)
    print(predict_nums[i])
    cv2.putText(card,predict_nums[i],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),3)

cv2.imshow('card', card)
cv2.waitKey(0)