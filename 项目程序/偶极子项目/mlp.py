import torch.nn as nn
import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d, UpsamplingNearest2d, Dropout2d,Flatten,Linear


class mlp_model(nn.Module):
    def __init__(self):
        super(mlp_model, self).__init__()
        self.Linear1 = Linear(21*21,128)
        self.Linear2 = Linear(128, 2)
        self.relu = ReLU()

        # Dropout Layer
        self.Drop = Dropout2d(p=0.1)

        self.flatten = Flatten()


    def forward(self, x):
        x = self.flatten(x)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Drop(x)
        x = self.Linear2(x)



        return x



if __name__ == '__main__':
    x1 = torch.rand([1, 1, 21, 21])
    label1 = torch.rand([1, 2])
    models = mlp_model()
    import time
    from thop import profile

    start = time.time()
    flops, params = profile(models, inputs=(x1,))
    end = time.time()
    print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    print(round((end-start)*1000,2) ,'ms')
