"""衍射loss和衍射网络数据两个文件夹，文件夹里面分别放入数字对应的文件，loss用txt保存"""
import os
import numpy as np
import PIL.Image as Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd
import matplotlib.pyplot as plt

'''路径部分'''
txt_path = r'./data.txt'
img_path = r'F:\Result\onn数字\onn级联unet数据sigmoid输出'
img_extension = '.png'
label_path = r'F:/衍射网络级联unet/all_image/label'
file_list = os.listdir(img_path)

def ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim

'''统计图片集的平均指标'''
with open(txt_path, 'a') as f1:
    f1.truncate(0)
    f1.writelines('layer_number distance loss loss_onn ssim psnr')
    f1.writelines('\n')


file_list.sort()
for num in file_list:
    img_num_path = os.path.join(img_path, num)
    ssim_num = 0
    psnr_num = 0
    img_list = os.listdir(img_num_path)
    for img_name in img_list:
        filename, extension = os.path.splitext(img_name)
        if extension != img_extension:
            continue
        s = img_name.split(',')
        ss = s[0].split('_')
        img = os.path.join(img_num_path, img_name)
        # label_img = os.path.join(label_path, s[2]) # imagenet
        label_img = os.path.join(label_path,ss[1],s[1]) #数字


        im1 = Image.open(img).convert('L')
        im1 = np.array(im1)
        Label = Image.open(label_img).convert('L')
        Label = Label.resize((im1.shape[0], im1.shape[1]))
        Label = np.array(Label)
        ssim_num += compare_ssim(im1, Label)

        psnr_num += compare_psnr(im1, Label)
    ssim_avg = round(ssim_num / len(img_list), 4)
    psnr_avg = round(psnr_num / len(img_list), 4)
    Result = f'{num} {ssim_avg} {psnr_avg}'
    print(Result)
    with open(txt_path,'a') as f1:
        f1.writelines(Result)
        f1.writelines('\n')

data = pd.read_table(txt_path, sep=' ')
dataFrame = pd.DataFrame(data)
with pd.ExcelWriter('./data.xlsx') as writer:
    dataFrame.to_excel(writer,sheet_name='onn级联unet',float_format='%.6f')
print('done')