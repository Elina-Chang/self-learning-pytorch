import torch
import numpy as np

data=[[1,2],[3,4]]
x_data=torch.tensor(data)

np_array=np.array(data)
x_np=torch.from_numpy(np_array)

x_ones=torch.ones_like(x_data)
x_rand=torch.rand_like(x_data,dtype=torch.float)

shape=(2,3,)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)

tensor=torch.rand(3,4)

if torch.cuda.is_available():
    tensor=tensor.to("cuda")

tensor=torch.ones(4,4)
tensor[:,1]=0

t1=torch.cat([tensor,tensor,tensor],dim=1)

y1=tensor@tensor.T
y2=tensor.matmul(tensor.T)
y3=torch.rand_like(y1)
torch.matmul(tensor,tensor.T,out=y3)

z1=tensor*tensor
z2=tensor.mul(tensor)
z3=torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)

agg=tensor.sum()
agg_item=agg.item()

tensor.add_(5)

t=torch.ones(5)
n=t.numpy()
t.add_(1)

n=np.ones(5)
t=torch.from_numpy(n)

np.add(n,1,out=n)

