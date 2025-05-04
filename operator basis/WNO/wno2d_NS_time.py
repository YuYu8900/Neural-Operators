"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 2-D Darcy equation (time-independent problem).
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
from wavelet_convolution_v3 import WaveConv2d

# torch.manual_seed(0)
# np.random.seed(0)
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# %%
""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, omega, padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
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
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2d(self.width, self.width, self.level, size=self.size, 
                                         wavelet=self.wavelet, omega=self.omega) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.mish( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
# %%
""" Model configurations """
TRAIN_PATH = 'ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'ns_data_V100_N1000_T50_2.mat'

ntrain = 1000
ntest = 200

sub = 1
S = 64
T_in = 10
T = 10
step = 1

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 4        # lavel of wavelet decomposition
width = 30       # uplifting dimension
layers = 4       # no of wavelet layers

grid_range = [1, 1]          # The grid boundary in x and y direction
in_channel = 12   # (a(x, y), x, y) for this case

# %%
""" Read data """
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

X = torch.tensor(np.linspace(0, 1, S), dtype=torch.float).reshape(1, S)
Y = torch.tensor(np.linspace(0, 1, S), dtype=torch.float).reshape(1, S)

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = WNO2d(width=width, level=level, layers=layers, size=[S,S], wavelet=wavelet,
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
    train_l2_step = 0
    train_l2_batch = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        
        for t in range(0, T, step):
            y = yy[..., t:t + step] # t:t+step, retains the third dimension,

            im = model(xx)            
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        train_l2_batch += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_batch = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_batch += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    train_loss[ep] = train_l2_step/ntrain/(T/step)
    test_loss[ep] = test_l2_step/ntest/(T/step)
    
    t2 = default_timer()
    scheduler.step()
    print('Epoch-{}, Time-{:0.4f}, Train-L2-Batch-{:0.4f}, Train-L2-Step-{:0.4f}, Test-L2-Batch-{:0.4f}, Test-L2-Step-{:0.4f}'
          .format(ep, t2-t1, train_l2_step/ntrain/(T/step), train_l2_batch/ntrain, test_l2_step/ntest/(T/step),
          test_l2_batch/ntest))

# %%
""" Prediction """
prediction = []
test_e = []     
with torch.no_grad():
    
    index = 0
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_batch = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        prediction.append( pred.cpu() )
        test_l2_step += loss.item()
        test_l2_batch += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        test_e.append( test_l2_step )
        index += 1
        
        print("Batch-{}, Test-loss-step-{:0.6f}, Test-loss-batch-{:0.6f}".format(
            index, test_l2_step/batch_size/(T/step), test_l2_batch) )
        
prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))         
print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/batch_size/(T/step), '%')
