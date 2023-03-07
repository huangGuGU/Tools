import torch
import torch.nn as nn
from PIL import Image  # 导入图片处理工具
import PIL.ImageOps
import numpy as np
from torchvision import transforms
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt



# 定义神经网络
class Hao(nn.Module):
    def __init__(self):
        super(Hao, self).__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数（即用了几个卷积核），第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10,kernel_size=(5,5))  # 因为图像为黑白的，所以输入通道为1,此时输出数据大小变为28-5+1=24.所以batchx1x28x28 -> batchx10x24x24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5,5))  # 第一个卷积层的输出通道数等于第二个卷积层是输入通道数。
        # self.conv2_drop = nn.Dropout2d()  # 在前向传播时，让某个神经元的激活值以一定的概率p停止工作，可以使模型泛化性更强，因为它不会太依赖某些局部的特征
        self.fc1 = nn.Linear(320, 50)  # 由于下部分前向传播处理后，输出数据为20x4x4=320，传递给全连接层。# 输入通道数是320，输出通道数是50
        self.fc2 = nn.Linear(50, 10)  # 输入通道数是50，输出通道数是10，（即10分类（数字1-9），最后结果需要分类为几个就是几个输出通道数）。全连接层（Linear）：y=x乘A的转置+b


    def forward(self, x):
        a =x
        x1 = self.conv1(x)
        x2 = torch.max_pool2d(x1, 2)
        x3 = F.relu(x2)  # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半，步长为2）（激活函数ReLU不改变形状）
        # x = F.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)),2))  # 此时输出数据大小变为12-5+1=8（卷积核大小为5）（2*2的池化层会减半）。所以 batchx10x12x12 -> batchx20x4x4。
        x4 = self.conv2(x3)
        x5 = torch.max_pool2d(x4,2)
        x6 = F.relu(x5)

        x7 = x6.view(-1, 320)  # batch*20*4*4 -> batch*320
        x8 = F.relu(self.fc1(x7))  # 进入全连接层
        # x = F.dropout(x, training=self.training)  # 减少遇到过拟合问题，dropout层是一个很好的规范模型。
        x9 = self.fc2(x8)
        out = F.softmax(x9,dim=1)
        return out

    # def __init__(self):
    #     super(Hao, self).__init__()
    #     self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
    #         nn.Conv2d(
    #             in_channels=1,  # 输入通道数
    #             out_channels=16,  # 输出通道数
    #             kernel_size=5,  # 卷积核大小
    #             stride=1,  # 卷积步数
    #             padding=2,  # 如果想要 con2d 出来的图片长宽没有变化,
    #             # padding=(kernel_size-1)/2 当 stride=1
    #         ),  # output shape (16, 28, 28)
    #         nn.ReLU(),  # activation
    #         nn.MaxPool2d(kernel_size=2),
    #
    #     )# 在 2x2 空间里向下采样, output shape (16, 14, 14) )
    #     self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
    #         nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
    #         nn.ReLU(),  # activation
    #         nn.MaxPool2d(2),  # output shape (32, 7, 7) )
    #     )
    #     self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层，0-9一共10个类
    #     # 前向反馈
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
    #     output = self.out(x)
    #     return output


