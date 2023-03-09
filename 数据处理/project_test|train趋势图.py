#####################################################################################
# 需要填入train和test的loss数据
# 数据类型是用：分割 epoch：loss：accuracy
# train_loss和test_loss分别是两个txt文件
# Range是个list，如果有时候loss改变太大，最后几个epoch趋势看不到，那就切片[a,b]范围的loss趋势
#####################################################################################
from 装饰器.decorator_程序启动 import logit
import matplotlib.pyplot as plt


@logit
def draw_loss_curve(train_loss, test_loss, save_path, Range):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    with open(train_loss) as f:
        for c in f.readlines():
            s = c.split(':')
            loss = float(s[1][:6])
            epoch = s[0]
            x_train.append(epoch)
            y_train.append(loss)

    with open(test_loss) as f1:
        for c in f1.readlines():
            s = c.split(':')
            loss = float(s[1][:6])
            epoch = s[0]
            x_test.append(epoch)
            y_test.append(loss)

        n = range(len(x_train))
        plt.plot(n, y_train, label='CNN loss')
        plt.plot(n, y_test, label='MLP loss')
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(save_path + '20 loss.png')

        plt.figure()
        n = range(Range[0], Range[1])
        plt.plot(n, y_train[Range[0]:Range[1]], label='CNN loss')
        plt.plot(n, y_test[Range[0]:Range[1]], label='MLP loss')
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(save_path + f'20 loss_{Range[0]}~{Range[1]}.png')


if __name__ == '__main__':
    train_loss = r''
    test_loss = r''
    save_path = r'/Users/hzh/Desktop/'
    Range = [700, 1500]
    draw_loss_curve(train_loss, test_loss, save_path, Range)
