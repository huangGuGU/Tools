import cv2
import os

ratio = 1148 / 860
path = r'E:\download\test\Test'
target =r'E:\download\test\choice'
if os.path.exists(target):
    pass
else:
    os.mkdir(target)

img_list = os.listdir(path)
try:
    img_list.remove('.DS_Store')
except:
    pass

for image in img_list:
    img_path = os.path.join(path, image)
    target_path = os.path.join(target, image)

    img = cv2.imread(img_path, 0)
    w, h = img.shape()
    Ratio = w / h
    if Ratio > ratio - 0.5 and Ratio < ratio + 0.5:
        cv2.imwrite(target_path, img)

print('done')


# import numpy as np
# import os
# import PIL.Image as Image
#
# ratio = 1148 / 860
# path = r'E:\download\test\Test'
# target =r'E:\download\test\choice'
# if os.path.exists(target):
#     pass
# else:
#     os.mkdir(target)
#
# img_list = os.listdir(path)
# try:
#     img_list.remove('.DS_Store')
# except:
#     pass
#
# for image in img_list:
#     img_path = os.path.join(path, image)
#     target_path = os.path.join(target, image)
#
#     Img = Image.open(img_path)
#     img = np.array(Img)
#     w = img.shape[0]
#     h = img.shape[1]
#
#     Ratio = w / h
#     if Ratio > ratio - 0.5 and Ratio < ratio + 0.5:
#         Img.save(target_path)
#
# print('done')

