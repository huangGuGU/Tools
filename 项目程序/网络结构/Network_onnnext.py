import torch.nn as nn
import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d, Softmax, UpsamplingNearest2d, Dropout2d





class SelfAttention(nn.Module):

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.snconv_qurey = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.snconv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.snconv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.snconv_attention = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.snmaxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps [B  C  W  H]
            returns :
                attention: [B  C  N] (N is Width*Height)
                out : input feature + (value @ attention)
        """
        b, c, h, w = x.size()

        qurey = self.snconv_qurey(x)
        qurey = qurey.view(-1, c // 8, h * w)

        key = self.snconv_key(x)
        key = self.snmaxpool(key)
        key = key.view(-1, c // 8, (h // 2) * (w // 2))

        attention = self.softmax(torch.bmm(qurey.permute(0, 2, 1), key))

        value = self.snconv_value(x)
        value = self.snmaxpool(value)
        value = value.view(-1, c // 2, (h // 2) * (w // 2))

        a = torch.bmm(value, attention.permute(0, 2, 1))
        a = a.view(-1, c // 2, h, w)
        a = self.snconv_attention(a)

        out = x + self.sigma * a
        return out


class onn_next(nn.Module):
    def __init__(self):
        super(onn_next, self).__init__()

        self.Conv1 = Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding='same')
        self.Conv2 = Conv2d(32, 64, kernel_size=(7, 7), stride=(1, 1), padding='same')
        self.Conv3 = Conv2d(64, 128, kernel_size=(7, 7), stride=(1, 1), padding='same')
        self.Conv4 = Conv2d(128, 64, kernel_size=(7, 7), stride=(1, 1), padding='same')
        self.Conv5 = Conv2d(128, 64, kernel_size=(7, 7), stride=(1, 1), padding='same')
        self.Conv6 = Conv2d(64, 32, kernel_size=(7, 7), stride=(1, 1), padding='same')
        self.Conv7 = Conv2d(32, 1, kernel_size=(7, 7), stride=(1, 1), padding='same')
        # Max pooling
        self.Maxpool = MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.Maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Nearest Up sampling
        self.NNI = UpsamplingNearest2d(scale_factor=4)
        self.NNI2 = UpsamplingNearest2d(scale_factor=2)
        # Dropout Layer
        self.Drop = Dropout2d(p=0)
        # Batch Norm Layer
        self.relu = ReLU()
        self.bn1 = BatchNorm2d(32)
        self.bn2 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.sa1 = SelfAttention(32)
        # self.sa2 = SelfAttention(128)

    def forward(self, x):
        x = self.Conv1(x)

        x_skip1 = x
        x = self.Maxpool2(x)
        x = self.sa1(x)
        x_skip2 = x
        x = self.Drop(x)
        x = self.relu(self.bn1(x))
        x = self.Maxpool2(x)

        x = self.Conv2(x)
        x_skip3 = x
        x = self.Drop(x)
        x = self.relu(self.bn2(x))
        x = self.Maxpool(x)

        x = self.Conv3(x)
        x = self.Drop(x)

        x = self.relu(self.bn3(x))
        # x = self.sa2(x)


        x = self.Conv4(x)

        x = self.Drop(x)
        x = self.relu(self.bn2(x))

        x = self.NNI(x)
        x = torch.cat([x, x_skip3], dim=1)
        x = self.Conv5(x)



        x = self.Drop(x)
        x = self.relu(self.bn2(x))
        x = self.NNI2(x)
        x = torch.cat([x, x_skip2], dim=1)
        x = self.Conv6(x)


        x = self.Drop(x)
        x = self.relu(self.bn1(x))
        x = self.NNI2(x)
        x = torch.cat([x, x_skip1], dim=1)
        x = self.Conv7(x)

        outputs = self.softmax(x)

        # outputs = torch.tanh(x)

        # outputs = self.sigmoid(x)

        return outputs


if __name__ == '__main__':
    x1 = torch.rand([4, 1, 240, 240])
    label1 = torch.rand([4, 1, 240, 240])
    models = onn_next()
    loss_function = torch.nn.MSELoss()
    prediction1 = models(x1)
    loss = loss_function(prediction1, label1)
    print(loss)
