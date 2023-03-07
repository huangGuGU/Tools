#####################################################################################
# 目标：把消化内镜书本批量转换为jpg后，将第六章页码上的每个图片截取出来，每一页做一个文件夹
# path 是书本的jpg格式文件夹的路径
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
            img_get = img[y + 5:y + h - 5, x + 5:x + w - 5]  # 对原图延伸5个像素点
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
    img_list = sorted(img_list, key=lambda x: int(x.split('_')[-1][:-4]))

    class_dic = {151: 'C3', 152: '-', 153: 'x', 154: 'C3', 155: 'C3', 156: 'C3', 157: 'C3', 158: '-', 159: 'C3',
                 160: 'C3', 161: 'C3', 162: '-', 163: 'C3', 164: '-', 165: 'C3', 166: 'C3', 167: '-', 168: 'C3',
                 169: 'C3', 170: 'C3', 171: 'C3', 172: 'C3', 173: 'C3', 174: 'C3', 175: 'C3', 176: '-', 177: 'C3',
                 178: 'C3', 179: 'C3', 180: 'C3', 181: 'C3', 182: 'C3', 183: 'C3', 184: '-', 185: 'C3', 186: 'C3',
                 187: '-', 188: 'C3', 189: 'C3', 190: '-', 191: 'C3', 192: 'C3', 193: 'C3', 194: 'C3', 195: 'C3',
                 196: '-', 197: 'C3', 198: 'C3', 199: 'C3', 200: 'C3', 201: '-', 202: 'C3', 203: '-', 204: 'C3',
                 205: 'C3', 206: 'C3', 207: 'C4.1', 208: 'C3', 209: '-', 210: 'C3', 211: '-', 212: 'C4.1', 213: '-',
                 214: 'C4.1', 215: 'C4.1', 216: 'C4.1', 217: 'C4.1', 218: '-', 219: 'C4.4', 220: '-', 221: 'C4.4',
                 222: '-', 223: 'C4.4', 224: '-', 225: 'C4.4', 226: 'C4.4', 227: '-', 228: 'C4.4', 229: '-',
                 230: 'C4.4', 231: '-', 232: 'C4.4', 233: 'C4.4', 234: 'C4.4', 235: '-', 236: 'C5', 237: 'C5',
                 238: '-', 239: 'C5', 240: '-', 241: 'C5', 242: 'C5', 243: '-', 244: 'C5', 245: '-', 246: 'C5',
                 247: '-', 248: 'C5', 249: '-', 250: 'C5', 251: '-', 252: 'C5'}  # 页码和对应Vienna

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
            print(f'{page_number} exclude')
    print('all done')


if __name__ == '__main__':
    path = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/内窥镜/肠道书和图片/下消化第六章图'
    save = r'/Users/hzh/Downloads/处理文件夹'
    cut_photo(path, save)
