
import cv2
import os

path = '/Users/hzh/Desktop/img'
save = '/Users/hzh/Desktop/save'

file_list = os.listdir(path)
try:
    file_list.remove('.DS_Store')
except:
    pass


def process(img_path, save_path):
    img = cv2.imread(img_path)
    img_filp = cv2.flip(img, 1)  # 镜像
    cv2.imwrite(save_path, img_filp)


for img_file in file_list:
    Img_path = os.path.join(path, img_file)
    Save_path = os.path.join(save, img_file)

    if not os.path.exists(Save_path):
        os.makedirs(Save_path)

    img_list = os.listdir(Img_path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass

    for img in img_list:
        img_path = os.path.join(Img_path, img)
        save_path = os.path.join(Save_path, img)
        process(img_path, save_path)

print('done')
