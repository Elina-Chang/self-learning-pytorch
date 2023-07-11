import os
import torch
from torch import nn
from torchvision.utils.data import DataLoader
from torchvision import datasets,transforms

# get decive for training
device=("cuda" if torch.cuda.is_available() else "mps" if torch.backends.map.is_available else "cpu")
print(f"using {device} device")

# define the model class
class Network(nn.Module):
    def __init__(self,in_channel=28*28,out_channel=10):
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(nn.Linear(in_features=in_channel,out_features=512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,out_features=out_channel))
    def forward(self,X):
        X=self.flatten(X)
        logits=self.linear_relu_stack(X)
        return logits

# create an instance of Network, move it to the device, print its structure
model=Network().to(device=device)
print(model)

# input and using the model
X=torch.rand(1,28,28,device=device)
logits=model(X)
# do not call model.forward directly
pred_probab=nn.Softmax(dim=1)(logits)
y_pred=pred_probab.argmax(1)

# more layers
input=torch.rand(3,28,28)


# 定义一个sequential，也可以直接传input进去
seq_modules=nn.Sequential(
          nn.Flatten(),
          nn.Linear(28*28,20),
          nn.ReLU(),
          nn.Linear(20,10),
        )
logits=seq_modules(input)
logits=nn.Softmax(dim=1)(logits)
# TO SCALE THE LOGITS TO [0,1], TO MATCH THE LABEL CLASS