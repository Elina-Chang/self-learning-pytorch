import torch
import torchvision.models as models


model=model.vgg16(weights="")
torch.save(model.state_dict(),"model_weights.pth")