import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# DATA
training_data=datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
test_data=datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())

# PLOT
label_map={0:"t-shirt",1:"trouser",2:"pullover",3:"dress",4:"coat",5:"sandal",6:"shirt",7:"sneaker",8:"bag",9:"ankel boot"}
figure=plt.figure(figsize=(8,8))
cols,rows=3,3
for i in range(1,cols*rows+1):
    sample_idx
    img,label=training_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(label_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()