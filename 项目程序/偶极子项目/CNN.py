import torch.nn as nn
import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d, UpsamplingNearest2d, Dropout2d,Flatten,Linear


class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()
        self.Conv1 = Conv2d(1, 16, kernel_size=(3, 3))
        self.Conv2 = Conv2d(16, 32, kernel_size=(3, 3))
        self.Conv3 = Conv2d(32, 64, kernel_size=(3, 3))
        self.maxpooling = MaxPool2d(kernel_size=(2,2))
        # Dropout Layer
        self.Drop = Dropout2d(p=0.1)
        # Batch Norm Layer
        self.relu = ReLU()
        self.bn1 = BatchNorm2d(16)
        self.bn2 = BatchNorm2d(32)
        self.bn3 = BatchNorm2d(64)

        self.flatten = Flatten()

        self.fc1 = Linear(64,10)
        self.fc2 = Linear(10, 2)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.Drop(x)

        x = self.Conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.Drop(x)

        x = self.Conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.Drop(x)

        x = self.flatten(x)
        x = self.fc1(x)
        outputs = self.fc2(x)





        return outputs



if __name__ == '__main__':
    from thop import profile
    import time
    x1 = torch.rand([1, 1, 21, 21])
    label1 = torch.rand([1, 2])
    models = cnn_model()


    start = time.time()
    flops, params = profile(models, inputs=(x1,))
    end = time.time()
    print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print(round((end-start)*1000,2) ,'ms')
    # loss_function = torch.nn.MSELoss()
    # prediction1 = models(x1)
    # loss = loss_function(prediction1, label1)
    # print(loss)
