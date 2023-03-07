#########################################################################
# txt_path 是存放指标的路径（为了写入excel准备，是个中间变量，不用管）
# img_path和label_path就是结果和labe的路径，遍历两个路径图片，计算指标
# img_extension就是图片后缀，.jpg或者.png
#########################################################################
from 装饰器.decorator_程序启动 import logit
import os
import numpy as np
import PIL.Image as Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd
from medpy import metric


@logit
def polyp_count(txt_path, excel_path, img_path, label_path, img_extension):
    file_list = os.listdir(img_path)

    with open(txt_path, 'a') as f1:
        f1.truncate(0)
        f1.writelines('dataset SSIM PSNR dice mIoU accuracy Recall Precision')
        f1.writelines('\n')

    for dataset in file_list:
        img_num_path = os.path.join(img_path, dataset)
        [SSIM_num, PSNR_num, dice, mIoU, acc, recall, sensitivity, specificity, precision] = [0] * 9

        img_list = os.listdir(img_num_path)
        for img_name in img_list:
            filename, extension = os.path.splitext(img_name)
            if extension != img_extension:
                continue
            img = os.path.join(img_num_path, img_name)
            # label_img = os.path.join(label_path, s[2]) # imagenet
            label_img = os.path.join(label_path, dataset, 'masks', img_name)  # 数字

            im1 = Image.open(img).convert('L')
            im1 = np.array(im1)
            Label = Image.open(label_img).convert('L')
            Label = Label.resize((im1.shape[1], im1.shape[0]), Image.BILINEAR)
            Label = np.array(Label)

            SSIM_num += compare_ssim(im1, Label)
            PSNR_num += compare_psnr(im1, Label)
            dice += metric.dc(im1, Label)
            mIoU += metric.jc(im1, Label)
            precision += metric.precision(im1, Label)
            recall += metric.recall(im1, Label)
            sensitivity += metric.sensitivity(im1, Label)
            specificity += metric.specificity(im1, Label)

        SSIM_avg = round(SSIM_num / len(img_list), 4)
        PSNR_avg = round(PSNR_num / len(img_list), 4)
        dice_avg = round(dice / len(img_list), 4)
        mIoU_avg = round(mIoU / len(img_list), 4)
        acc_avg = round(acc / len(img_list), 4)
        recall_avg = round(recall / len(img_list), 4)
        precision_avg = round(precision / len(img_list), 4)

        Result = f'{dataset} {SSIM_avg} {PSNR_avg} {dice_avg} {mIoU_avg} {acc_avg} {recall_avg} {precision_avg}'
        print(Result)

        with open(txt_path, 'a') as f1:
            f1.writelines(Result)
            f1.writelines('\n')

    data = pd.read_table(txt_path, sep=' ')
    data_frame_polyp = pd.DataFrame(data)
    with pd.ExcelWriter(excel_path) as writer:
        data_frame_polyp.to_excel(writer, sheet_name='息肉数据', float_format='%.6f')
    print('done')


if __name__ == '__main__':
    txt_path = r'./data.txt'
    excel_path = r'./data.xlsx'

    img_path = r'..\results\HarDMSEG'
    label_path = r'..\dataset\TestDataset'
    img_extension = '.png'
    polyp_count(txt_path, excel_path, img_path, label_path, img_extension)
