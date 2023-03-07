import os
import numpy as np
import PIL.Image as Image

path = r''
save = r''
num_list = os.listdir(path)
try:
    num_list.remove('.DS_Store')
except:
    pass

'''获取train_img中某个数字类文件夹的某个数字'''
for num in num_list:
    num_path = os.path.join(path, num)
    save_num_path = os.path.join(save, num)
    img_list = os.listdir(num_path)
    for image in img_list:
        img = os.path.join(num_path, image)
        save_img = os.path.join(save_num_path, image)
        img = Image.open(img).convert('L')
        im = np.array(img)

        '''计算左右上线的界限   a1 = 左界限  a2 = 右界限  b1 = 上界限  b2 = 下界限 '''
        v1 = v2 = v3 = v4 = 0
        a1 = a2 = b1 = b2 = 0
        m, n = im.shape
        for i in range(0, m, 1):
            for j in range(0, n, 1):
                if im[i, j] >= 1:
                    b1 = i
                    v1 = 1
                    break
            if v1 == 1:
                break
        for i in range(0, n, 1):
            for j in range(0, m, 1):
                if im[j, i] >= 1:
                    a1 = i
                    v2 = 1
                    break
            if v2 == 1:
                break
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if im[i, j] >= 1:
                    b2 = i
                    v3 = 1
                    break
            if v3 == 1:
                break
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if im[j, i] >= 1:
                    a2 = i
                    v4 = 1
                    break
            if v4 == 1:
                break

        L1 = a2 - a1  # 左右的长度
        L2 = b2 - b1  # 上下的长度
        img_cut = im[ b1:b2,a1:a2]
        im = Image.fromarray(img_cut).convert('L')
        if os.path.exists(save_num_path) == False:
            os.makedirs(save_num_path)
        im.save(save_img)