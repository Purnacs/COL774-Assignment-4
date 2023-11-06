import torch.nn as nn
import torch
import os
import torchvision 

def convolution(x):
    l1 = nn.Conv1d(3,32,5)(x)
    l1_rel = nn.ReLU()(l1)
    l1_pool = nn.AvgPool1d(2)(l1_rel)
    l2 = nn.Conv1d(32,64,5)(l1_pool)
    l2_rel = nn.ReLU()(l2)
    l2_pool = nn.AvgPool1d(2)(l2_rel)
    l3 = nn.Conv1d(64,128,5)(l2_pool)
    l3_rel = nn.ReLU()(l3)
    l3_pool = nn.AvgPool1d(2)(l3_rel)
    l4 = nn.Conv1d(128,256,5)(l3_pool)
    l4_rel = nn.ReLU()(l4)
    l4_pool = nn.AvgPool1d(2)(l4_rel)
    l5 = nn.Conv1d(256,512,5)(l4_pool)
    l5_rel = nn.ReLU()(l5)
    result = nn.AvgPool2d(3)(l5_rel)
    return result

def import_data(file):
    os.listdir(file)
    pass
