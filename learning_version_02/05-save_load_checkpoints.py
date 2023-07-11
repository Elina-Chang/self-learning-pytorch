import torch
import torch.nn as nn
import torch.optim as optim


model=Net()
optim.SGD(model.parameters(),lr=0.1,momentum=0.1)
torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"loss":loss},save_path)
# load checkpoints
checkpoint=torch.load(path)
# 太差劲了，神马东西
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()