import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img_name = 'TestImage_18'
path = '/Users/hzh/Desktop/output'
target = '/Users/hzh/Desktop/label'
save = '/Users/hzh/Desktop/result'
color_flag = False

if not os.path.exists(save):
    os.mkdir(save)
img_list = os.listdir(path)
label_list = os.listdir(target)
try:
    img_list.remove('.DS_Store')
    label_list.remove('.DS_Store')
except:
    pass
for i in img_list:
    img_path = os.path.join(path, i)
    target_path = os.path.join(target, i)
    save_path = os.path.join(save, i)

    if color_flag:
        im1 = cv.imread(img_path)
        im2 = cv.imread(target_path)
        im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)
        im2 = cv.cvtColor(im2, cv.COLOR_BGR2RGB)
        im1 = np.float32(im1 / 255)
        im2 = np.float32(im2 / 255)
        Different_point = np.where(im1 != im2)
        im1[Different_point[0][:],Different_point[1][:],2] = 0
        im1[Different_point[0][:], Different_point[1][:], 1] = 255
        im1[Different_point[0][:], Different_point[1][:], 0] = 0
        plt.imshow(im1)
        plt.axis('off')
        plt.show()

        # plt.savefig(save_path)



    else:
        Im1 = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        Im2 = cv.imread(target_path, cv.IMREAD_GRAYSCALE)
        # Im1 = np.float32(Im1 / 255)
        # Im2 = np.float32(Im2 / 255)
        Im3 = np.abs(Im1 - Im2)
        print(Im3.max())



    plt.imshow(Im3)
    plt.axis('off')
    plt.show()


    # save_img = save_path+'.png'
    # cv.imwrite(save_img, im3)

print('done')








