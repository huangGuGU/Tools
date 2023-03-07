import os
import cv2 as cv

path = '/Users/hzh/Desktop/x'
target = '/Users/hzh/Desktop/y'
if os.path.exists(target) == False:
    os.makedirs(target)
img_list = os.listdir(path)
try:
    img_list.remove('.DS_Store')
except:
    pass

for img in img_list:
    img_path = os.path.join(path, img)
    target_path = os.path.join(target, img)
    img = cv.imread(img_path, -1)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    cv.imwrite(target_path, thresh)
    pass

