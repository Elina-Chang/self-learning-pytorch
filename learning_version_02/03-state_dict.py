import torch

"""
In PyTorch, the learnable parameters (i.e. weights and biases) of a torch.nn.Module model are contained in the model’s parameters (accessed with model.parameters()). 
A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor.
"""
for param_tensor in net.state_dict():
    pass
for var_name in optimizer.state_dict():
    pass