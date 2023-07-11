# A rundown of an easy project using pytorch
# 1. working with data
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# data section
training_data=datasets.FashionMNIST(root="data",train=True,transform=ToTensor(),download=True)
test_data=datasets.FashionMNIST(root="data",train=False,transform=ToTensor(),download=True)

# dataloader section
batch_size=64
train_dataloader=DataLoader(dataset=training_data,batch_size=batch_size)
test_dataloader=DataLoader(dataset=test_data,batch_size=batch_size)

# model section
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(nn.Linear(28*28,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,10))
    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits
# 实例化model, put model on device
device=("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model=Network().to(device)

# loss & optimizer
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=model.parameters,lr=1e-3)

# train section: eg. one epoch
def train(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        pred=model(X)
        loss=loss_fn(pred,y)
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print loss
        if batch % 100 == 0:
            loss,current=loss.item(),(batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# test section
def test(dataloader,model,loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for X,y in enumerate(dataloader):
            X,y=X.to(device),y.to(device)
            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss/=num_batches
    correct/=size
    # print loss & accuracy
    pritn(f"Test Error: \n Accuracy: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f} \n")

# loop for multiple epochs
epochs=5
from tqdm import tqdm, trange
for t in trange(epochs):
    print(f"Epoch {t+1}\n---------------------------------------")
    train(dataloader=train_dataloader,model=model,loss_fn=loss_fn,optimizer=optimizer)
    test(dataloader=test_dataloader,model=model,loss_fn=loss_fn)
print("DONE!")

# save model
torch.save(obj=model.state_dict(),f="model.pth")
print("Save model state fo model.pth")

# loading model
model=Network().to(device)
model.load_state_dict(state_dict=torch.load("model.pth"))

# application
classes=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
model.eval()
X,y=test_data[0][0],test_data[0][1]
with torch.no_grad():
    pred=model(X)
    predicted,actual=classes[pred[0].argmax(0)],classes[y]
    print(f"Predicted: {predicted}, Actual: {actual}")