import torch
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda

target_transform=Lambda(lambda y:torch.zeros(10,dtype=torch.float).scatter_(dim=0,index=torch.tensor(y),value=1))
ds=datasets.FashionMNIST(root="data",train=True,transform=ToTensor(),target_transform=target_transform,download=True)