import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image 
import cv2
import torch
import os
import sys
import torchvision 

def convolution(x: torch.Tensor) -> torch.Tensor:
    results = []
    for i in x:
        l1 = nn.Conv2d(3,32,5)(i)
        l1_rel = nn.ReLU()(l1)
        l1_pool = nn.AvgPool1d(2)(l1_rel)
        l2 = nn.Conv2d(32,64,5)(l1_pool)
        l2_rel = nn.ReLU()(l2)
        l2_pool = nn.AvgPool1d(2)(l2_rel)
        l3 = nn.Conv2d(64,128,5)(l2_pool)
        l3_rel = nn.ReLU()(l3)
        l3_pool = nn.AvgPool1d(2)(l3_rel)
        l4 = nn.Conv2d(128,256,5)(l3_pool)
        l4_rel = nn.ReLU()(l4)
        l4_pool = nn.AvgPool1d(2)(l4_rel)
        l5 = nn.Conv2d(256,512,5)(l4_pool)
        l5_rel = nn.ReLU()(l5)
        result = nn.AvgPool2d(3)(l5_rel)
        results.append(result)
    result = torch.stack(results)
    return result

def import_data(dir_path,type:str): # import data -> NOTE: sys.argv[1] would be path to "Dataset" , so the dir_name passed should be sys.argv[1] + "/SyntheticData/images" etc
    '''
    Imports data from the given directory path and returns a pandas dataframe and torch.Tensor
    Args:
        dir_path: path to the folder containing the csv and images
        type: type of data to be imported -> "train" or "test" or "val"
    Returns:
        i) pandas dataframe containing image name and corresponding output
        ii) input for convolutional function -> torch.Tensor

    Workflow:
    1) Read CSV file and convert to pandas dataframe of image name and corresponding output
    2) From image names in the dataframe, read images from the directory (specifically images would be in ./images folder of pwd)
    3) Convert images to torch.Tensor
    4) Return dataframe and torch.Tensor
    '''
    df = pd.read_csv(dir_path + "/SyntheticData/" + type + ".csv")  # read csv file
    df["image"] = df["image"].apply(lambda x: dir_path + "/SyntheticData/images/" + x)  # add path to image name
    images = []  # list to store images
    #Read images -> NOTE: images are read as torch.Tensor, resized to 224x224 and normalised'
    #NOTE: torchvision.transforms.ToTensor() converts the image to torch.Tensor 
    #NOTE: torchvision.transforms.Resize() resizes the image to the given size
    #NOTE: torchvision.transforms.Normalize() normalises the image over the given mean and standard deviation
    #NOTE: torchvision.transforms.Normalize() expects the image to be of shape (C,H,W) where C is the number of channels'
    #NOTE: torchvision.transforms.Normalize() expects the mean and standard deviation to be of shape (C) where C is the number of channels'
    #NOTE: torchvision.transforms.Normalize() expects the mean and standard deviation to be of type list or tuple'
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((224,224)),torchvision.transforms.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])])
    n_images = 0
    for i in df["image"]:
        image = cv2.imread(i) # read image
        #Normalise over image
        images.append(transform(image)) # append image to list
        n_images+=1
        if n_images == 32:
            break
    images = torch.stack(images) # convert list of torch.Tensor to torch.Tensor
    return df,images

'''Testing Area'''
if __name__ == '__main__':
    dir_path = sys.argv[1]
    # dirname = os.path.dirname(__file__)
    # dir_path = os.path.join(dirname,"../Dataset")
    df,images = import_data(dir_path,"train")
    # print(df)
    # print(images)
    print(convolution(images))