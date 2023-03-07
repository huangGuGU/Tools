import os
import cv2 as cv
path = '/Users/hzh/Desktop/x'
target = '/Users/hzh/Desktop/y'
img_list = os.listdir(path)
try:
    img_list.remove('.DS_Store')
except:
    pass

for img in img_list:
    img_path = os.path.join(path, img)
    target_path = os.path.join(target, img)



#########################################################################################################################

    img = cv.imread(img_path)
    img_filp = cv.flip(img, 1)  # 镜像
    '''
    参数2 必选参数。用于指定镜像翻转的类型，其中0表示绕×轴正直翻转，即垂直镜像翻转；1表示绕y轴翻转，即水平镜像翻转；-1表示绕×轴、y轴两个轴翻转，即对角镜像翻转。
    参数3 可选参数。用于设置输出数组，即镜像翻转后的图像数据，默认为与输入图像数组大小和类型都相同的数组。
    '''
    cv.imwrite(target_path, img_filp)

print('done')