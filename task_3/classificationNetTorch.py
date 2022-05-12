import torch
from torch import nn
from torch.nn import init


class ClassificationNet(torch.nn.Module):

    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.linear1 = torch.nn.Linear(3000, 1000, dtype=torch.double)
        self.dropout = torch.nn.Dropout(p=0.6)
        self.linear2 = torch.nn.Linear(1000, 1, dtype=torch.double)
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
