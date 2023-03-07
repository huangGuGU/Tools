#####################################################################################
# 仅适用于全光网络的权重绘制
# 使用时先查看odic的key的名字，然后手动放到sheet_name中
# path就是全光权重的路径
# save就是excel保存的路径
#####################################################################################
import torch
from matplotlib import pyplot as plt
from 装饰器.decorator_程序启动 import logit
import os
import pandas as pd


@logit
def draw_weight(path, save):
    if os.path.exists(save):
        pass
    else:
        os.mkdir(save)
    lam = 632e-9

    weights = torch.load(path, map_location=torch.device('cpu')).items()

    d_list = []
    name_list = []
    for name, weight in weights:
        name_list.append(name)
        if 'phase_modulator' in name:
            phi = torch.nn.Sigmoid()(weight) * 2 * torch.pi
            # height = (phi * lam / ((1.55 - 1) * 2 * torch.pi)).numpy()
        else:
            phi = weight
            # height = (phi * lam / ((1.55 - 1) * 2 * torch.pi)).numpy()

        # d_list.append(height)
        d_list.append(phi)

        # plt.imshow(phi, cmap='gray')
        # plt.axis('off')
        # # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig('/Users/hzh/Desktop/x' + '//' + name + '.png',bbox_inches='tight',pad_inches=0.0)
        # plt.show()
    dataFrame0 = pd.DataFrame(d_list[0])
    dataFrame1 = pd.DataFrame(d_list[1])
    dataFrame2 = pd.DataFrame(d_list[2])

    with pd.ExcelWriter(f'{save}/L1.xlsx') as writer:
        dataFrame0.to_excel(writer, sheet_name=f'L1', float_format='%.9f', header=False, index=False)

    with pd.ExcelWriter(f'{save}/L2.xlsx') as writer:
        dataFrame1.to_excel(writer, sheet_name=f'L2', float_format='%.9f', header=False, index=False)

    with pd.ExcelWriter(f'{save}/diffuser_L0.xlsx') as writer:
        dataFrame2.to_excel(writer,sheet_name=f'diffuser_L0', float_format='%.9f', header=False, index=False)

    df0 = pd.read_excel(f'{save}/L1.xlsx')
    df0.to_csv(f'{save}/L1.txt', sep='	', float_format='%.9f', header=False, index=False)

    df1 = pd.read_excel(f'{save}/L2.xlsx')
    df1.to_csv(f'{save}/L2.txt', sep='	', float_format='%.9f', header=False, index=False)

    df2 = pd.read_excel(f'{save}/diffuser_L0.xlsx')
    df2.to_csv(f'{save}/diffuser_L0.txt', sep='	', float_format='%.9f', header=False, index=False)


if __name__ == '__main__':
    path = r'/Users/hzh/Desktop/Onn_Net_70_-0.937341.pth'
    save = r'/Users/hzh/Desktop/相位'

    draw_weight(path, save)
