#####################################################################################
# 目标：把消化内镜书本批量转换为jpg后，将上消化道章页码上的每个图片截取出来，每一页做一个文件夹
# path是书本的jpg格式文件夹的路径
# save是保存的文件夹的路径
# 字典中-代表不是想要的图片,x代表没有说明类型
#####################################################################################
from 装饰器.decorator_程序启动 import logit
import os
import cv2


def Process(imgCopy):
    img = 255 - cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    return img


def getContours(img, imgProcess):
    img_list = []
    contours, hierarchy = cv2.findContours(imgProcess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000 * 1000:  # 面积大于像素记为封闭图形
            peri = cv2.arcLength(cnt, True)  # 计算周长
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 计算有多少个拐角
            x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小
            img_get = img[y + 5:y + h - 5, x + 5:x + w - 5]  # 对原图延伸50个像素点
            img_list.append(img_get)
    return img_list


def cut_save(img_path, save_file, page_number):
    img = cv2.imread(img_path)
    img_process = Process(img)
    img_list = getContours(img, img_process)

    for index, img in enumerate(img_list):
        save_path = os.path.join(save_file, str(index) + '.jpg')
        w, h, c = img.shape
        if abs(w - h) < 300:
            cv2.imwrite(save_path, img)
    print(f'{page_number} done')


@logit
def cut_photo(path, save):
    img_list = os.listdir(path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass

    img_list = sorted(img_list, key=lambda x: int(x.split('_')[-1][:-4]))  # 按照页码先从小到大排序

    class_dic = {40: 'C4.2', 41: 'C4.2', 42: '-', 43: 'C4.2', 44: '-', 45: 'C4.4', 46: 'C4.4', 47: 'C4.4', 48: 'C5',
                 49: 'C5', 50: 'C4.4', 51: '-', 52: 'C4.2', 53: '-', 54: 'C4.4', 55: 'C4.4', 56: '-', 57: 'C4.4',
                 58: '-', 59: 'C4.4', 60: 'C4.4', 61: '-', 73: 'C4.4', 74: '-', 75: 'C3', 76: '-', 77: 'C3', 78: '-',
                 79: 'C3', 80: 'C4.1', 81: '-', 82: 'C4.1', 83: 'C4.1', 84: 'C4.2', 85: '-', 86: 'C4.4', 87: '-',
                 88: 'C4.4', 89: 'C4.4', 90: '-', 91: 'C4.4', 92: '-', 93: 'C4.4', 94: 'C4.4', 95: 'C4.4', 96: 'C4.4',
                 97: 'C4.4', 98: 'C4.4', 99: 'C4.4', 100: '-', 101: 'C4.4', 102: '-', 103: 'C4.4', 104: 'C4.4',
                 105: 'C4.4', 106: 'C4.4', 107: 'C4.4', 108: 'C4.4', 109: 'C4.4', 110: '-', 111: 'C4.4', 112: '-',
                 113: 'C4.4', 114: 'C4.4', 115: 'C4.4', 116: 'C4.4', 117: 'C5', 118: 'C5', 119: '-', 120: 'C4'}
    # 页码和对应Vienna

    for index, image in enumerate(img_list):
        page_number = int(image.split('_')[-1][:-4]) - 11
        img_path = os.path.join(path, image)
        vienna_class = class_dic[page_number]
        if vienna_class != 'x' and vienna_class != '-':
            save_file = os.path.join(save, 'Page_' + str(page_number) + f'_{vienna_class}')
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            cut_save(img_path, save_file, page_number)
        else:
            print(f'{page_number} --------------- exclude ---------------')
    print('all done')


if __name__ == '__main__':
    path = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/内窥镜/肠道书和图片/上消化道图'
    save = r'/Users/hzh/Downloads/上消化道处理文件夹'
    cut_photo(path, save)
