"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 1-D Burger's equation (time-independent problem).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution_v3 import WaveConv1d

torch.manual_seed(0)
np.random.seed(0)
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# %%
""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, omega, padding=0):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: scalar (for 1D), right support of 1D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 2: (a(x), x)
        for i in range( self.layers ):
            self.conv.append( WaveConv1d(self.width, self.width, self.level, size=self.size,
                                         wavelet=self.wavelet, omega=self.omega) )
            self.w.append( nn.Conv1d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)              # Shape: Batch * x * Channel
        x = x.permute(0, 2, 1)       # Shape: Batch * Channel * x
        if self.padding != 0:
            x = F.pad(x, [0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:   # Final layer has no activation    
                x = F.mish(x)        # Shape: Batch * Channel * x 
                
        if self.padding != 0:
            x = x[..., :-self.padding] 
        x = x.permute(0, 2, 1)       # Shape: Batch * x * Channel
        x = F.mish( self.fc1(x) )    # Shape: Batch * x * Channel
        x = self.fc2(x)              # Shape: Batch * x * Channel
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%
""" Model configurations """
Ntotal = 2000
ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 3       # lavel of wavelet decomposition
width = 64       # uplifting dimension
layers = 4       # no of wavelet layers

h = 972 # total grid size divided by the subsampling rate
grid_range = 1
in_channel = 3   # (a(x), x) for this case

# %%
""" Read data """

PATH_Sigma = 'elasticity/Meshes/Random_UnitCell_sigma_10.npy'
PATH_XY = 'elasticity/Meshes/Random_UnitCell_XY_10.npy'

input_s = np.load(PATH_Sigma)
input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
input_xy = np.load(PATH_XY)
input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)

train_s = input_s[:ntrain]
test_s = input_s[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]

print(train_s.shape, train_xy.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_xy, train_s), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_xy, test_s), batch_size=batch_size,
                                          shuffle=False)

# %%
""" The model definition """
model = WNO1d(width=width, level=level, layers=layers, size=h, wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range, omega=2).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward() # l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest
    
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2

    t2 = default_timer()
    print('Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}'
          .format(ep, t2-t1, train_mse, train_l2, test_l2))

# %%
""" Prediction """
pred = []
test_e = []
with torch.no_grad():
    
    index = 0
    for x, y in test_loader:
        test_l2 = 0 
        x, y = x.to(device), y.to(device)

        out = model(x)
        test_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_e.append( test_l2/batch_size )
        pred.append( out )
        print("Batch-{}, Test-loss-{:0.6f}".format( index, test_l2/batch_size ))
        index += 1

pred = torch.cat((pred))
test_e = torch.tensor((test_e))  
print('Mean Error:', 100*torch.mean(test_e).numpy())

