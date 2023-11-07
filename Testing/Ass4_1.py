import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image 
import cv2
import torch
import os
import sys
import torchvision 

class CNN:
    def __init__(self):
        self.layer1 = nn.Conv2d(3,32,5)
        self.layer2 = nn.Conv2d(32,64,5)
        self.layer3 = nn.Conv2d(64,128,5)
        self.layer4 = nn.Conv2d(128,256,5)
        self.layer5 = nn.Conv2d(256,512,5)

        self.pooling = nn.AvgPool2d(2)
        self.relu = nn.ReLU()
        
    def single_example_cnn(self,x):
        x = self.layer1(x)
        x = self.pooling(self.relu(x))
        x = self.layer2(x)
        x = self.pooling(self.relu(x))
        x = self.layer3(x)
        x = self.pooling(self.relu(x))
        x = self.layer4(x)
        x = self.pooling(self.relu(x))
        x = self.layer5(x)
        return self.pooling(self.relu(x))
    
    def fit_cnn(self,x):
        results = []
        for i in x:
            results.append(self.single_example_cnn(i))
        return torch.stack(results)


def import_data(dir_path,is_synthetic:bool = True,type:str = "train"): # import data -> NOTE: sys.argv[1] would be path to "Dataset" , so the dir_name passed should be sys.argv[1] + "/SyntheticData/images" etc
    '''
    Imports data from the given directory path and returns a pandas dataframe and torch.Tensor
    Args:
        dir_path: path to the folder containing the csv and images
        is_synthetic: Synthetic/Handwritten Data
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
    if is_synthetic:
        data_path = "/SyntheticData/"
        image_path = "images/"
    else:
        data_path = "/HandwrittenData/"
        image_path = "images/" + type + "/"
        type += "_hw"
    df = pd.read_csv(dir_path + data_path + type + ".csv")  # read csv file
    df["image"] = df["image"].apply(lambda x: dir_path + data_path + image_path + x)  # add path to image name
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
    df,images = import_data(dir_path,True,"train")
    # print(df)
    # print(images)
    cnn = CNN()
    print(cnn.fit_cnn(images))