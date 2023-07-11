import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,in_channel):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=32,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
        self.dropout1=nn.Dropout2d(p=0.25)
        self.dropout2=nn.Dropout2d(p=0.5)
        self.fc1=nn.Linear(in_features=9216,out_features=128)
        self.fc2=nn.Linear(in_features=128,out_features=10)
    def __forward__(self,X):
        X=self.conv1(X)
        ReLU()
        

model=Net()