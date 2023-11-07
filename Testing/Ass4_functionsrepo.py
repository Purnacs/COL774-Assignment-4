import torch.nn as nn
import torch

def convolution(x: torch.Tensor) -> torch.Tensor:
    results = []
    for i in x:
        l1 = nn.Conv2d(3,32,5)(i)
        l1_rel = nn.ReLU()(l1)
        l1_pool = nn.AvgPool2d(2)(l1_rel)
        l2 = nn.Conv2d(32,64,5)(l1_pool)
        l2_rel = nn.ReLU()(l2)
        l2_pool = nn.AvgPool2d(2)(l2_rel)
        l3 = nn.Conv2d(64,128,5)(l2_pool)
        l3_rel = nn.ReLU()(l3)
        l3_pool = nn.AvgPool2d(2)(l3_rel)
        l4 = nn.Conv2d(128,256,5)(l3_pool)
        l4_rel = nn.ReLU()(l4)
        l4_pool = nn.AvgPool2d(2)(l4_rel)
        l5 = nn.Conv2d(256,512,5)(l4_pool)
        l5_rel = nn.ReLU()(l5)
        result = nn.AvgPool2d(3)(l5_rel)
        results.append(result)
    result = torch.stack(results)
    return result