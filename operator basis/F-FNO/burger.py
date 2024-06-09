import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from matplotlib.pyplot import flag
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utilities3 import *
from linear import WNLinear
from feedforward import FeedForward
from timeit import default_timer



class SpectralConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, modes, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes
        self.mode = mode
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        if self.mode != 'no-fourier':
            x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b m i -> b i m')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M = x.shape

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_ftx.new_zeros(B, I, M// 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :,  :self.modes] = torch.einsum(
                "bix,iox->box",
                x_ftx[:, :,  :self.modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.modes] = x_ftx[:, :, :, :self.modes]

        x = torch.fft.irfft(out_ft, n=M, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        x = rearrange(x, 'b i m -> b m i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class FNOFactorizedMesh1D(nn.Module):
    def __init__(self, modes, width, input_dim, n_layers, share_weight, factor,
                 ff_weight_norm, n_ff_layers, layer_norm):
        super().__init__()
        self.padding = 2  # pad the domain if input is non-periodic
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.in_proj = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.n_layers = n_layers

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes]:
                weight = torch.FloatTensor(width, width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv1d(in_dim=width,
                                                       out_dim=width,
                                                       modes=modes,
                                                       forecast_ff=None,
                                                       backcast_ff=None,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       use_fork=False,
                                                       dropout=0.0,
                                                       mode='full'))

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, 1, wnorm=ff_weight_norm))

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # [B, X, Y, 4]
        x = self.in_proj(x)  # [B, X, H]
        x = x.permute(0, 2, 1)  # [B, H, X]
        # x = F.pad(x, [0, self.padding, 0, self.padding])
        x = x.permute(0, 2, 1)  # [B, X, H]

        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            b, _ = layer(x)
            x = x + b

        # b = b[..., :-self.padding, :-self.padding, :]
        output = self.out(b)

        return output

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

ntrain = 1000
ntest = 100

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

batch_size = 100
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes = 16
width = 64
input_dim = 2
n_layers = 24#4
share_weight = 'false'
factor = 2  
ff_weight_norm ='true'
n_ff_layers = 1
layer_norm = 'false'

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('/home/yy/yy/data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
model = FNOFactorizedMesh1D(modes, width, input_dim, n_layers, share_weight, factor,
                 ff_weight_norm, n_ff_layers, layer_norm).cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
