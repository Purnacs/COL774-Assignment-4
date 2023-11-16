import torch
import os
import cv2
import pandas as pd
import numpy as np
import torch.nn as nn
from torchvision import transforms
import sys
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LatexDataset(Dataset):
    def __init__(self,df):
        self.df = df
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])])
    
    def __getitem__(self,index):
        row = self.df.iloc[index]
        img_path = row['image']
        img = Image.open(img_path)
        img = self.transform(img)
        return img,row['formula']
    
    def __len__(self):
        return len(self.df)

# TODO: Could update this function to be a bit more better
def get_path(dir_path,is_synthetic,data_type):
    if is_synthetic: 
        data_path = "/SyntheticData/"
        image_path = "images/"
    else:
        data_path = "/HandwrittenData/"
        image_path = "images/" + data_type + "/"
        data_type += "_hw"
    csv_path = dir_path + data_path + data_type + ".csv"
    img_path = dir_path + data_path + image_path
    return csv_path,img_path

def import_data(dir_path, is_synthetic:bool=True, data_type:str = "train"):
    '''
    Imports data from the given directory and uses the LatexDataset class and the torch.DataLoader to create a dataloader iterator
    Args:
        dir_path: The parent directory path to dataset (specifically ./Dataset)
        is_synthetic: Variable for determining whether to use synthetic or Handwritten Data
        data_type: Variable for determining train, test or validation dataset to return
    Returns:
        A torch.DataLoader object containing the required Data ]in batches with shuffling(if training data)]
    '''
    batch_size = 64
    is_shuffle = True if data_type == "train" else False

    # The below code block is only because of the given dataset directory structure -> Should probably make this a function on its own for easy updates
    csv_path,img_path = get_path(dir_path,is_synthetic,data_type)

    df = pd.read_csv(csv_path) # read csv
    df["image"] = df["image"].apply(lambda x: img_path + x) # add path to image name
    dataset = LatexDataset(df)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=is_shuffle,num_workers=4)
    return dataloader

'''
Using the above functions for data_processing:
Check out the following link for more info -> https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''

if __name__ == '__main__':

    #NOTE: Use the following small codeblock for device usage (particularly when using HPC)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    dir_path = sys.argv[1]
    train_syn = import_data(dir_path,True,"train")
    test_syn = import_data(dir_path,True,"test")
    val_syn = import_data(dir_path,True,"val")
    train_hw = import_data(dir_path,False,"train")
    val_hw = import_data(dir_path,False,"val")

    # print(next(iter(train_syn))) # -> Convert the DataLoader object to iterator and then call next for getting the batches one by one which would contain tensor of both images and formulas

    print("Successful loading of all data")

    '''
    # NOTE: Each batch would be a 4D Tensor of size [batch_size, num_channels, height, width], whereas each image within it is a 3D tensor of size [num_channels, height, width]
    
    A Sample code for using the dataloader assuming some model
    model = ...
    criterion = ...
    optimizer = ...
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to GPU if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss for every 100 steps
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')
    '''