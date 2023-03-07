#####################################################################################
#
# 衍射loss和衍射网络数据两个文件夹，文件夹里面分别放入数字对应的文件，loss用txt保存
#
#####################################################################################
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr



'''路径部分'''
# loss_path = '/Users/hzh/衍射网络Data/衍射loss'
img_path = '/Users/hzh/Desktop/空域/incoherent'
# save_loss_path = '/Users/hzh/衍射网络Data/衍射loss趋势图'
img_extension = '.png'
label_path = '/Users/hzh/Desktop/label'

# loss_list = os.listdir(loss_path)
file_list = os.listdir(img_path)
try:
    file_list.remove('.DS_Store')
    # loss_list.remove('.DS_Store')
except:
    pass

'''画出loss趋势图到save_loss_path'''
# for loss in loss_list:
#     Loss_path = os.path.join(loss_path, loss)
#     name, extension = os.path.splitext(loss)
#     x = []
#     y = []
#     with open(Loss_path) as f:
#         for c in f.readlines():
#             s = c.split(':')
#             loss = float(s[1][:6])
#             epoch = s[0]
#             x.append(epoch)
#             y.append(loss)
#
#     n = range(len(x))
#     plt.plot(n, y, label='loss')
#     plt.margins(0)
#     plt.subplots_adjust(bottom=0.15)
#     plt.xlabel("epoch")
#     plt.ylabel("loss")  # Y轴标签
#     plt.title(f"{name}_loss")  # 标题
#
#     Save_path = f"{save_loss_path}/{name}_loss.png"
#     plt.savefig(Save_path)
#     plt.clf()

'''统计图片集的平均指标'''
file_list.sort()
for num in file_list:
    img_num_path = os.path.join(img_path, num)
    ssim_num = 0
    psnr_num = 0
    loss_num = 0
    img_list = os.listdir(img_num_path)
    try:
        img_list.remove('.DS_Store')
    except:
        pass
    for img_name in img_list:
        filename, extension = os.path.splitext(img_name)
        if extension != img_extension:
            continue
        s = img_name.split(',')
        img = os.path.join(img_num_path, img_name)
        label_img = os.path.join(label_path, s[-1])
        loss = float(s[0].split('_')[0])

        im1 = Image.open(img).convert('L')
        im1 = np.array(im1)
        Label = Image.open(label_img).convert('L')
        Label = Label.resize((im1.shape[0], im1.shape[1]))
        Label = np.array(Label)
        ssim_num += compare_ssim(im1, Label)
        psnr_num += compare_psnr(im1, Label)
        loss_num += loss
    ssim_avg = round(ssim_num / len(img_list), 4)
    psnr_avg = round(psnr_num / len(img_list), 4)
    loss_avg = round(loss_num / len(img_list), 4)
    print(f'{num} : SSIM: {ssim_avg}, PSNR: {psnr_avg}, NPCC:{loss_avg}')
