import torch

x=torch.ones(5)
y=torch.zeros(3)
w=torch.rand(5,3,requires_grad=True)
b=torch.rand(3,requires_grad=True)
z=torch.matmul(x,w)+b

loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
# set the value of requires_grad when creating a tensor, or later by using x.requires_grad_(True)
loss.backward() # 反传，计算参数梯度只需这一句话
print(w.grad)
print(b.grad)

# By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation. 
z=torch.matmul(x,w)+b
print(z.requires_grad)
# i.e. we only want to do forward computations through the network when we have trained the model and just want to apply it to some input data
with torch.no_grad(): # surrounding our computation code with torch.no_grad() block
    z=torch.matmul(x,w)+b
print(z.requires_grad)
# DETACH METHOD
z=torch.matmul(x,w)+b
z_det=z.detach()
print(z_det.requires_grad)
z.detach_()
print(z.requires_grad)