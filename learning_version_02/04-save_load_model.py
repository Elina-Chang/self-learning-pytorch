import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def __forward__(self,X):
        pass
    return predicted_vector

optimizer=optim.SGD(Net().parameters(),lr=0.1,momentum=0.9)

# 两种
# 1. model.state_dict
torch.save(Net().state_dict(),save_path=".pt/.pth") 
# corresponding load: 
Net().load_state_dict(torch.load(save_path)), Net().eval()
# 2. model self
torch.save(Net(),save_path=".pt/.pth") 
# corresponding load: 
model=torch.load(save_path), model.eval()
"""
Notice that the load_state_dict() function takes a dictionary object, NOT a path to a saved object. 
This means that you must deserialize the saved state_dict before you pass it to the load_state_dict() function. 
For example, you CANNOT load using model.load_state_dict(PATH).
"""
"""
Remember too, that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. 
Failing to do this will yield inconsistent inference results.
"""
model.eval()