import torch
import torch.utils.data.dataset
from torch.utils.data import Dataset
import numpy as np

label_list = []

for i in range(1000):
    label_list.append(f'{i}_{i % 2}')
label_list = np.array(label_list)


class MyDataSet(Dataset):
    def __init__(self, label_list):
        self.input = []
        self.label = []

        for i in label_list:
            self.input.append(float(i.split('_')[0]))
            self.label.append(float(i.split('_')[1]))

    def __getitem__(self, index):
        input = torch.tensor(self.input[index])
        label = torch.tensor(self.label[index])
        return input, label

    def __len__(self):
        return len(self.input)


def dataloader():
    trainset = MyDataSet(label_list)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=0)
    testset = MyDataSet(label_list)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader


train_loader, test_loader = dataloader()


class MyModel(torch.nn.Module):
    def __init__(self,):
        super(MyModel, self).__init__()
        self.linear1 = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(5, 10)
        self.linear3 = torch.nn.Linear(10, 5)
        self.linear4 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = 0.5 * torch.cos(torch.pi * (1 - x)) + 0.5
        return x


net = MyModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
epoch_list = []
loss_list = []
if __name__ == '__main__':
    for epoch in range(100000):
        correct = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.unsqueeze(1)

            y_pred = net(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%50==0:
            print(loss.item())
