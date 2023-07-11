import os
import pandas as pd
from torchvision.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform,target_transform):
        self.img_labels=pd.read_csv(annotations_file)
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image=read_image(img_path)
        label=self.img_labels.iloc[idx,1]
        if self.transform:
            image=self.transform(image)
        if self.target_transform:
            label=self.target_transform(label)
        return image,label

# 熟练背诵：
# the __getitem__ function loads and returns a sample from the dataset at the given index idx. based on the index, it identifies the image's location on disk, 
# convets that to a tensor using read_image, retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicabel),
# and returns the tensor image and corresponding label in a tuple.
# 重点：返回的image和label是在一个tuple里