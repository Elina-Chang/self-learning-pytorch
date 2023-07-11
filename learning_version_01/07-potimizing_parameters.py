import torch
from torch import nn
from torchvision.utils.data import Dataset
from torchvision.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# hyper parameters
batch_size=64
epoch=200
learning_rate=1e-3

# 两个关键的东西: loss, optimizer
# 反传计算梯度的时候
loss.backward()
optimizer.step()
optimizer.zero_grad() # 为了方式梯度累加, 每个epoch后都要执行这个操作
## 其实就是四步: 放进dataloader(dataset), 实例化model并计算predict=model(input), 计算loss(pred,label)并梯度反传.backward(), 优化参数optimizer.step()
# dataloader 要传batch_size, optimizer 要传learning_rate
epochs=10
for t in range(epochs):
    train_loop(dataLoader,model,loss_fn,optimizer)
    test_loop(dataloader,model,loss_fn)

# train_loop里面包含了从dataloader里面取出来的batch, 隔多少个batch print一下loss
# test_loop里面要把梯度去掉，with torch.no_grad():, 这里可能要加上出loss以外的评价指标了, 如accuracy