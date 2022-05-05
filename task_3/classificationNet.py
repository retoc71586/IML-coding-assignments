import torch
from torch import nn
from torch.nn import init


class ClassificationNet(torch.nn.Module):

    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.linearFirst = torch.nn.Linear(3000, 1000, dtype=torch.double)
        self.dropout = torch.nn.Dropout(p=0.7)
        self.linear1 = torch.nn.Linear(1000, 288, dtype=torch.double)
        self.linear2 = torch.nn.Linear(288, 72, dtype=torch.double)
        self.linear3 = torch.nn.Linear(72, 18, dtype=torch.double)
        self.linear4 = torch.nn.Linear(18, 1, dtype=torch.double)
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.linearFirst(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.sigmoid(x)
        return x
