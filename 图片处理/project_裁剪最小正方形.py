import os
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

path = r''
save = r''
padding = True
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
    try:
        img_list.remove('.DS_Store')
    except:
        pass
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
        L3 = max(L1, L2)
        center_length = a1 + (L1) / 2
        center_hight = b1 + (L2) / 2

        x1 = center_hight - L3 / 2
        y1 = center_length - L3 / 2
        x2 = int(x1 + L3)
        y2 = int(y1 + L3)

        img_cut = im[int(x1):x2 + 1, int(y1):y2 + 1]  #此处为最小矩形

        if padding:
            img_cut = transforms.ToTensor()(img_cut)
            img_padding = F.pad(img_cut, (2, 2, 2, 2))
            img_padding = torch.unsqueeze(img_padding,dim=0)
            im_resize = F.interpolate(img_padding, size=(160,160), mode='bilinear')
            img_padding2 = F.pad(im_resize, (40, 40, 40, 40))#(左边填充数， 右边填充数， 上边填充数， 下边填充数)
            img_padding2 = img_padding2[0,0, :, :].numpy() * 255
            im = Image.fromarray(img_padding2).convert('L')
        else:
            im = Image.fromarray(img_cut)
            
        if not os.path.exists(save_num_path):
            os.makedirs(save_num_path)
        im.save(save_img)
print('done')

