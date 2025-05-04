import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from timeit import default_timer
from pytorch_wavelets import DWT1D, IDWT1D


from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import module
from module import ConvAttention1, ConvAttention2, PreNorm, PreNorm2, FeedForward
from torch.optim import lr_scheduler as lr_scheduler
from trgcvt1D_module import Transformer1, Transformer2, CvTencoderdecoder1, CvTencoderdecoder2

torch.manual_seed(0)
np.random.seed(0)


# device= torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1')
#%%
class WaveConv1dtransformer(nn.Module):
    def __init__(self, in_channels, out_channels, level,dummy):
        super(WaveConv1dtransformer, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        """
        # print(dummy.shape)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        
        self.dwt_ = DWT1D(wave='db6', J=self.level, mode='symmetric').to(dummy.device)
        
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        
        self.transformer_1 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_2 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_3 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_4 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        
        
        # self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        # self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        # self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        
               
     # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self,x_1,x_2):
        #Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT1D(wave='db6', J=self.level, mode='symmetric').cuda()
        x_ft1, x_coeff1 = dwt(x_1) 
        x_ft2, x_coeff2 = dwt(x_2) 
        
        
         # Multiply the final low pass and high pass coefficients
        # out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        # out_ft[:, :, :self.modes1] = self.mul1d(x_ft[:, :, :self.modes1], self.weights1)
        # x_coeff[-1] = self.mul1d(x_coeff[-1][:, :, :self.modes1], self.weights1)

        


        x_ft= self.transformer_1(x_ft1.clone(),x_ft2.clone())
        # print(x_ft1.shape)
        # print(x_coeff2[-1][:,:,0,:,:].clone().shape)
        x_coeff1[-1] = self.transformer_2(x_coeff1[-1].clone(),x_coeff2[-1].clone())
        idwt = IDWT1D(mode='symmetric', wave='db6').cuda()
        x = idwt((x_ft, x_coeff1))
    
        return x
        
    

""" The forward operation """
class WNO1dtransformer(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO1dtransformer, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=S, y=S, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=S, y=S, c=1)
        """
        # print(dummy_data.shape)
        self.level = level
        self.dummy_data = dummy_data
        self.inp_size = dummy_data.shape[-1]
        self.width = width
        self.padding = 0 # pad the domain if input is non-periodic
        
        self.fc0 = nn.Linear(20, self.width) 
        self.conv0 = WaveConv1dtransformer(self.width, self.width, self.level, self.dummy_data)
        self.w0 = CvTencoderdecoder2(self.inp_size,self.width,self.width,self.width)
        # self.conv1 = WaveConv1dtransformer(self.width, self.width, self.level, self.dummy_data)
        # self.w1 = CvTencoderdecoder2(self.inp_size,self.width,self.width,self.width)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.w4 = nn.Conv2d(self.width, self.width, 1)
        # self.w5 = nn.Conv2d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        # self.fc1.weight = nn.Parameter(self.n1_ws,requires_grad=False)
        # self.fc1.bias = nn.Parameter(self.n1_bs,requires_grad=False)
        
        self.fc2 = nn.Linear(128, 1)
        # self.fc2.weight = nn.Parameter(self.n2_ws,requires_grad=False)
        # self.fc2.bias = nn.Parameter(self.n2_bs,requires_grad=False)
        
        

    def forward(self, x,y):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        y = torch.cat((y, grid), dim=-1)

        x = self.fc0(x)
        y = self.fc0(y)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # do padding, if required
        
        y = y.permute(0, 2, 1)
        # y = F.pad(y, [0,self.padding, 0,self.padding]) # do padding, if required
        
        x1 = self.conv0(x,y)
        x2 = self.w0(x,y)
       
        
        x = x1 + x2
        # x = F.gelu(x)

       
        # x = x[..., :-self.padding, :-self.padding] # remove padding, when required
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x   
    
    
    
    

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
