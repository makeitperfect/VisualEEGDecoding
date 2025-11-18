
import torch.nn as nn
import torch
import scipy.signal as signal
# from utils import *


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithAbs(nn.Conv2d):

    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithAbs, self).__init__(*args, **kwargs)

    def forward(self, x):
        
        if self.doWeightNorm: 
            # self.weight.data = torch.renorm(
            #     self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            # )

            self.weight.data = torch.abs(
                self.weight.data
            )
        return super(Conv2dWithAbs, self).forward(x)


class Brain_Visual_Encoder_EEG(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):

        super(Brain_Visual_Encoder_EEG, self).__init__()

        self.temporal_len   = temporal_len
        self.embed_channels = channels
        self.embed_dim      = 200

        self.eeg_encoder = nn.Sequential(
            nn.Linear(self.temporal_len,self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(self.embed_dim,   self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.65),
        )

        self.spatial_conv = Conv2dWithAbs(1,25,kernel_size=(channels,1),bias=False)
        self.bn = nn.BatchNorm2d(25)
    

        self.img_adapter = nn.Sequential(
            nn.ELU(),
            nn.Linear(proj_dim,1024),
            nn.ELU(),
            nn.Dropout(0.85),
            nn.Linear(1024,proj_dim),
        )


        self.eeg_adapter = nn.Sequential(
            nn.Linear(25*self.embed_dim,proj_dim),
        )


        self.learned_scale = nn.Parameter(torch.rand([1,50,proj_dim]),requires_grad = True)
        self.softplus      = nn.Softplus()


    def get_image_feature(self,imgs):
        rates = torch.softmax(self.learned_scale[:,:imgs.shape[1]],-2)
        img  = torch.sum(imgs * rates,1)
  
        img = self.img_adapter(img)
        return img

    def forward(self, x):
        x = self.spatial_conv(x[:,None])
        x = self.bn(x)

        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))

        return x

    
class Brain_Visual_Encoder_MEG(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):
        super(Brain_Visual_Encoder_MEG, self).__init__()

        self.temporal_len   = temporal_len
        self.embed_channels = channels
        self.embed_dim      = 150

        self.eeg_encoder = nn.Sequential(
            nn.Linear(self.temporal_len,self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(self.embed_dim,self.embed_dim),
            nn.ELU(),
            nn.Dropout(0.65),
        )


        self.spatial_conv = nn.Conv2d(1,50,kernel_size=(channels,1),bias=False)

        self.eeg_adapter = nn.Sequential(
            nn.Linear(50*self.embed_dim,proj_dim),
        )

        self.img_adapter = nn.Sequential(
            #nn.SiLU(),
            nn.ELU(),
            nn.Linear(proj_dim,1024),
            nn.Dropout(0.85),
            #nn.SiLU(),
            nn.ELU(),
            nn.Linear(1024,proj_dim),
        )

        self.learned_scale = nn.Parameter(torch.rand([1,50,proj_dim]),requires_grad = True)
 

    def get_image_feature(self,imgs):
  
        rates = torch.softmax(self.learned_scale[:,:imgs.shape[1]],-2)
        img  = torch.sum(imgs * rates,1)
  
        img = self.img_adapter(img)
        return img

    def forward(self, x):
        x = self.spatial_conv(x[:,None])
        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))


        return x
    



if __name__ == "__main__":

    x = torch.rand(3,17,250)
    net = Brain_Visual_Encoder_MEG(17,1024)

    img_1 = torch.rand(4,1,1024)
    img_2 = torch.rand(4,1,1024)
    imgs  = torch.cat([img_1,img_2],1)


    y = net(x)
    print(y.shape)