#####################################################################################
# path 是图片的文件夹路径
# box_path是边框文件夹路径，里面是xml文件
#####################################################################################
import os
import xml.etree.ElementTree as ET
import cv2


def find_box(filename):
    global x, y, X, Y, w, h
    tree = ET.parse(filename)
    root = tree.getroot()
    size = root.find('size')
    w = float(size.find('width').text)
    h = float(size.find('height').text)

    for node in root.iter("object"):
        bndbox = node.find('bndbox')
        x = int(bndbox.find('xmin').text)
        y = int(bndbox.find('ymin').text)
        X = int(bndbox.find('xmax').text)
        Y = int(bndbox.find('ymax').text)
    return x, y, X, Y, w, h


def draw_box():
    box_list = os.listdir(box_path)
    try:
        box_list.remove('.DS_Store')
    except:
        pass
    for index, box_file in enumerate(box_list):
        box_file_path = os.path.join(box_path, box_file)
        image_path = os.path.join(path, box_file)
        files = os.listdir(box_file_path)

        for f in files:
            x, y, X, Y, w, h = find_box(os.path.join(box_file_path, f))
            img_path = os.path.join(image_path, f[:-4] + '.JPEG')
            img = cv2.imread(img_path)
            img_box = cv2.rectangle(img, (x, y), (X, Y), (0, 255, 0), 2)
            cv2.imwrite(img_path, img_box)
        print(f'{index + 1}/{len(box_list)} done')

    print('all done')


if __name__ == '__main__':
    path = r''
    box_path = r''
    draw_box()
