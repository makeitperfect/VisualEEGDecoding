import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

'''
This code includes/modifies parts from the codebase of paper "Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior."

'''


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)
    
class EEGProjectLayer(nn.Module):
    def __init__(self, c_num, z_dim, timesteps, drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps)
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

        self.img_adapter = nn.Sequential(
            #nn.SiLU(),
            nn.ELU(),
            nn.Linear(z_dim,1024),
            nn.Dropout(0.85),
            #nn.SiLU(),
            nn.ELU(),
            nn.Linear(1024,z_dim),
        )

        self.learned_scale = nn.Parameter(torch.rand([1,50,z_dim]),requires_grad = True)


    def get_image_feature(self,imgs):
  
        rates = torch.softmax(self.learned_scale[:,:imgs.shape[1]],-2)
        img  = torch.sum(imgs * rates,1)
  
        img = self.img_adapter(img)
        return img
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x
    

class BaseModel(nn.Module):
    def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1440):
        super(BaseModel, self).__init__()
        
        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()


        self.img_adapter = nn.Sequential(
            #nn.SiLU(),
            nn.ELU(),
            nn.Linear(z_dim,1024),
            nn.Dropout(0.85),
            #nn.SiLU(),
            nn.ELU(),
            nn.Linear(1024,z_dim),
        )

        self.learned_scale = nn.Parameter(torch.rand([1,50,z_dim]),requires_grad = True)


    def get_image_feature(self,imgs):
  
        rates = torch.softmax(self.learned_scale[:,:imgs.shape[1]],-2)
        img  = torch.sum(imgs * rates,1)
  
        img = self.img_adapter(img)
        return img
    
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x


class Shallownet(BaseModel):
    def __init__(self,  c_num, z_dim, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1440)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
        
        self.c_num = c_num
        self.timesteps = timesteps
        # self.test_dim()


    def test_dim(self):

        x = torch.randn(1,1, self.c_num, self.timesteps[1] - self.timesteps[0])
        print('x shape:', x.shape)

        y = self.backbone(x)
        print('y shape:', y.shape,torch.flatten(y,1).shape[-1])
        self.embedding_dim = torch.flatten(y,1).shape[-1]
    
class Deepnet(BaseModel):
    def __init__(self,  c_num, z_dim, timesteps):
        super().__init__(z_dim, c_num, timesteps,embedding_dim = 1400)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )
        self.c_num = c_num
        self.timesteps = timesteps
        # self.test_dim()


    def test_dim(self):

        x = torch.randn(1,1, self.c_num, self.timesteps[1] - self.timesteps[0])
        print('x shape:', x.shape)

        y = self.backbone(x)
        print('y shape:', y.shape,torch.flatten(y,1).shape[-1])
        self.embedding_dim = torch.flatten(y,1).shape[-1]
      
class EEGnet(BaseModel):
    def __init__(self,  c_num, z_dim, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1248)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )
        self.c_num = c_num
        self.timesteps = timesteps
        # self.test_dim()


    def test_dim(self):

        x = torch.randn(1,1, self.c_num, self.timesteps[1] - self.timesteps[0])
        print('x shape:', x.shape)

        y = self.backbone(x)
        print('y shape:', y.shape,torch.flatten(y,1).shape[-1])
        self.embedding_dim = torch.flatten(y,1).shape[-1]

class TSconv(BaseModel):
    def __init__(self, c_num, z_dim, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1440)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
        self.c_num = c_num
        self.timesteps = timesteps
        # self.test_dim()


    def test_dim(self):

        x = torch.randn(1,1, self.c_num, self.timesteps[1] - self.timesteps[0])
        print('x shape:', x.shape)

        y = self.backbone(x)
        print('y shape:', y.shape,torch.flatten(y,1).shape[-1])
        self.embedding_dim = torch.flatten(y,1).shape[-1]



if __name__ == "__main__":
    # Example usage
    model = EEGProjectLayer(z_dim=128, c_num=17, timesteps=(0, 250))
    model = Shallownet(z_dim=128, c_num=64, timesteps=(0, 250))
    print(model)

    # Test the forward pass with dummy data
    x = torch.randn(10, 64, 250)  # Batch size of 10, 32 channels, 1000 time steps
    output = model(x)
    print(output.shape)  # Should be [10, 128]