#####################################################################################
# R是个list，如果有时候loss差别太大，最后几个epoch趋势看不到，查看[a,b]范围的loss趋势
# flag 是判断否切片
# txt格式  epoch：loss
#####################################################################################
from 装饰器.decorator_程序启动 import logit
import matplotlib.pyplot as plt


@logit
def draw_loss_curve(loss_list, name_list, save_path, Range, Flag, name):
    plt.figure(dpi=300)
    for index, L in enumerate(loss_list):
        train = []
        with open(L) as f:
            for c in f.readlines():
                s = c.split(':')
                loss = float(s[1][:6])
                train.append(loss)

        if Flag:
            n = range(Range[0], Range[1])
            plt.plot(n, train[Range[0]:Range[1]], label=name_list[index])
            name = f'{name} {Range[0]}~{Range[1]}'
        else:
            n = range(len(train))
            plt.plot(n, train, label=name_list[index])

    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(save_path + name + '.png')
    plt.show()


if __name__ == '__main__':
    loss1 = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/偶极子/Result/NoRing/选择超参数/CNN/sgd 1e-2/test_loss.txt'
    loss2 = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/偶极子/Result/NoRing/选择超参数/CNN/sgd 1e-3/test_loss.txt'
    loss3 = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/偶极子/Result/NoRing/选择超参数/CNN/adam 1e-2/test_loss.txt'
    loss4 = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/偶极子/Result/NoRing/选择超参数/CNN/adam 1e-3/test_loss.txt'
    loss5 = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/偶极子/Result/NoRing/选择超参数/CNN/rms 1e-2/test_loss.txt'
    loss6 = r'/Users/hzh/Library/Mobile Documents/com~apple~CloudDocs/工作/偶极子/Result/NoRing/选择超参数/CNN/rms 1e-3/test_loss.txt'

    # Loss_list = [loss1, loss2, loss3, loss4, loss5, loss6]
    # Name_list = ['SGD 1e-2', 'SGD 1e-3', 'Adam 1e-2', 'Adam 1e-3', 'RMS 1e-2', 'RMS 1e-3']
    Loss_list = [ loss5, loss6]
    Name_list = [ 'RMS 1e-2', 'RMS 1e-3']

    Save_path = r'/Users/hzh/Desktop/'
    Name = 'CNN loss'
    R = [1000, 1200]
    # flag = False
    flag = True
    draw_loss_curve(Loss_list, Name_list, Save_path, R, flag, Name)
