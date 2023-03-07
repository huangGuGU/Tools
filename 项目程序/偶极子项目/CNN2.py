import torch.nn as nn
import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d, UpsamplingNearest2d, Dropout2d,Flatten,Linear


class Cnn2_model(nn.Module):
    def __init__(self):
        super(Cnn2_model, self).__init__()



        self.Conv1 = Conv2d(1, 5, kernel_size=(5, 5),padding=(2,2))
        self.Conv2 = Conv2d(5, 5, kernel_size=(5, 5),padding=(2,2))

        self.maxpooling = MaxPool2d(kernel_size=(2,2))
        self.flatten = Flatten()


        # Dropout Layer
        self.Drop = Dropout2d(p=0.1)
        # Batch Norm Layer
        self.relu = ReLU()
        self.bn1 = BatchNorm2d(16)
        self.bn2 = BatchNorm2d(32)
        self.linear1 = Linear(125,100)
        self.linear2 = Linear(100, 2)





    def forward(self, x):
        x = self.Conv1(x)

        x = self.relu(x)
        x = self.maxpooling(x)


        x = self.Conv2(x)

        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.Drop(x)
        x = self.linear2(x)

        outputs = x





        return outputs



if __name__ == '__main__':
    x1 = torch.rand([64, 1, 21, 21])
    label1 = torch.rand([64, 2])
    models = Cnn2_model()
    loss_function = torch.nn.MSELoss()
    prediction1 = models(x1)
    loss = loss_function(prediction1, label1)
    print(loss)