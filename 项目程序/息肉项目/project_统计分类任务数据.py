#####################################################################################
# path是txt的路径，也就是网络结果保存 格式是 epoch:loss:accuracy
# 读取 loss和accuracy按照epoch画图
#####################################################################################
from 装饰器.decorator_程序启动 import logit
import matplotlib.pyplot as plt


@logit
def analyse_classify():
    loss_list = []
    accuracy_list = []
    epoch_list = []
    with open(path, 'r') as f:
        for Line in f.readlines():
            epoch_list.append(int(Line.split(':')[0][:5]))
            loss_list.append(float(Line.split(':')[1][:5]))
            accuracy_list.append(float(Line.split(':')[2][:5]))

    plt.plot(epoch_list, accuracy_list, label='accuracy')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy")
    plt.show()

    plt.plot(epoch_list, loss_list, label='loss')
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("loss")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = '/Users/hzh/Desktop/transformerU cifar10.txt'
    analyse_classify()
    pass
