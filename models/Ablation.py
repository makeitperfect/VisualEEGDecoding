
import torch.nn as nn
import torch
import scipy.signal as signal
# from utils import *

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)

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


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    设计巴特沃斯带通滤波器
    
    参数:
        lowcut: 通带低频截止频率(Hz)
        highcut: 通带高频截止频率(Hz)
        fs: 采样频率(Hz)
        order: 滤波器阶数
        
    返回:
        b, a: IIR滤波器的分子(b)和分母(a)多项式系数
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # print(type(b),type(a),type(data))
    y = signal.lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    """
    设计巴特沃斯高通滤波器
    
    参数:
        cutoff: 截止频率(Hz)
        fs: 采样频率(Hz)
        order: 滤波器阶数
        
    返回:
        b, a: IIR滤波器的分子(b)和分母(a)多项式系数
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    应用巴特沃斯高通滤波器
    
    参数:
        data: 输入信号
        cutoff: 截止频率(Hz)
        fs: 采样频率(Hz)
        order: 滤波器阶数
        
    返回:
        滤波后的信号
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y



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


class LinearWithAbs(nn.Linear):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithAbs, self).__init__(*args, **kwargs)

    def forward(self, x):
        
        if self.doWeightNorm: 
            self.weight.data = torch.abs(
                self.weight.data
            )
        return super(LinearWithAbs, self).forward(x)
    


class Brain_Visual_Encoder_EEG_wo_spatial(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):
        super(Brain_Visual_Encoder_EEG_wo_spatial, self).__init__()

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

        self.img_adapter = nn.Sequential(
            nn.ELU(),
            nn.Linear(proj_dim,1024),
            nn.ELU(),
            nn.Dropout(0.85),
            nn.Linear(1024,proj_dim),
        )

        self.eeg_adapter = nn.Sequential(
            nn.Linear(self.embed_channels*self.embed_dim,proj_dim),
        )

        self.learned_scale = nn.Parameter(torch.rand([1,50,proj_dim]),requires_grad = True)
        self.softplus      = nn.Softplus()


    def get_image_feature(self,imgs):
        rates = torch.softmax(self.learned_scale[:,:imgs.shape[1]],-2)
        img  = torch.sum(imgs * rates,1)
  
        img = self.img_adapter(img)
        return img

    def forward(self, x):
        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))

        return x


class Brain_Visual_Encoder_EEG_wo_feature_adapter(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):

        super(Brain_Visual_Encoder_EEG_wo_feature_adapter, self).__init__()

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
  
        # img = self.img_adapter(img)
        return img

    def forward(self, x):
        x = self.spatial_conv(x[:,None])
        x = self.bn(x)

        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))

        return x
    

class Brain_Visual_Encoder_EEG_wo_blur(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):

        super(Brain_Visual_Encoder_EEG_wo_blur, self).__init__()

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
        img = imgs[:,0]  
        img = self.img_adapter(img)
        return img
    

    def forward(self, x):
        x = self.spatial_conv(x[:,None])
        x = self.bn(x)

        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))

        return x


class Brain_Visual_Encoder_EEG_wo_blur_wo_feature_adapter(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):
        super(Brain_Visual_Encoder_EEG_wo_blur_wo_feature_adapter, self).__init__()


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
        img = imgs[:,0]  
        # img = self.img_adapter(img)
        return img
    

    def forward(self, x):
        x = self.spatial_conv(x[:,None])
        x = self.bn(x)

        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))

        return x


class Brain_Visual_Encoder_EEG_wo_spatial_wo_feature_adapter(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):
        super(Brain_Visual_Encoder_EEG_wo_spatial_wo_feature_adapter, self).__init__()

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

        self.img_adapter = nn.Sequential(
            nn.ELU(),
            nn.Linear(proj_dim,1024),
            nn.ELU(),
            nn.Dropout(0.85),
            nn.Linear(1024,proj_dim),
        )

        self.eeg_adapter = nn.Sequential(
            nn.Linear(self.embed_channels*self.embed_dim,proj_dim),
        )

        self.learned_scale = nn.Parameter(torch.rand([1,50,proj_dim]),requires_grad = True)
        self.softplus      = nn.Softplus()


    def get_image_feature(self,imgs):
        rates = torch.softmax(self.learned_scale[:,:imgs.shape[1]],-2)
        img  = torch.sum(imgs * rates,1)
  
        return img

    def forward(self, x):
        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))

        return x


class Brain_Visual_Encoder_EEG_wo_spatial_wo_blur(nn.Module):
    def __init__(self,channels = 63,proj_dim = 1152,temporal_len = 250):
        super(Brain_Visual_Encoder_EEG_wo_spatial_wo_blur, self).__init__()

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

        self.img_adapter = nn.Sequential(
            nn.ELU(),
            nn.Linear(proj_dim,1024),
            nn.ELU(),
            nn.Dropout(0.85),
            nn.Linear(1024,proj_dim),
        )

        self.eeg_adapter = nn.Sequential(
            nn.Linear(self.embed_channels*self.embed_dim,proj_dim),
        )

        self.learned_scale = nn.Parameter(torch.rand([1,50,proj_dim]),requires_grad = True)
        self.softplus      = nn.Softplus()


    def get_image_feature(self,imgs):
        img = imgs[:,0]  
        img = self.img_adapter(img)
        return img

    def forward(self, x):
        x = self.eeg_encoder(x)
        x = self.eeg_adapter(x.flatten(1))

        return x




if __name__ == "__main__":

    x = torch.rand(3,17,250)
    net = Brain_Visual_Encoder_EEG_wo_spatial_wo_blur(17,1024)

    img_1 = torch.rand(4,1,1024)
    img_2 = torch.rand(4,1,1024)
    imgs  = torch.cat([img_1,img_2],1)


    y = net(x)
    print(y.shape)