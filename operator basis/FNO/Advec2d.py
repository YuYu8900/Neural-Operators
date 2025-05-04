import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from matplotlib.pyplot import flag
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utilities3 import *
from timeit import default_timer


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2D(nn.Module):
    def __init__(self, modes1, modes2, width, n_layers):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)
        self.n_layers = n_layers

        self.convs = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        for _ in range(n_layers):
            conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            self.convs.append(conv)

            w = nn.Conv2d(self.width, self.width, 1)
            self.ws.append(w)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



ntrain = 1000
ntest = 100
nt = 40
nx = 40
s = 40

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

T = 40
step = 1

modes = 16
width = 32
n_layers = 4


data = np.load('time_advection/train_IC2.npz')
x, t, u_train = data["x"], data["t"], data["u"] 
u_train = torch.tensor(u_train, dtype=torch.float) # N x nt x nx
# x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
x_train = u_train[:, 0:1, :]  # N x nx
x_train = x_train.reshape(ntrain,nx,1,1).repeat([1,1,40,1])
# x = np.repeat(x[0:1, :], ntrain, axis=0)  # N x nx
# x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_train = u_train.permute(0, 2, 1)  # N x nx x nt
# x_train = torch.from_numpy(x_train)
# u_train = torch.from_numpy(u_train)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)

data = np.load('time_advection/test_IC2.npz')
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
u_test  = torch.tensor(u_test, dtype=torch.float)
x_test = u_test[:ntest, 0, :].reshape(ntest,nx ,1)   # N x nx
x_test  = x_test.reshape(ntest,nx,1,1).repeat([1,1,40,1])
# x = np.repeat(x[0:1, :], ntest, axis=0)  # N x nx
# x_test = np.concatenate((u0_test[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_test = u_test[:ntest, :, :].permute(0, 2, 1)  # N x nx x nt
# x_test = torch.from_numpy(x_test)
# u_test = torch.from_numpy(u_test)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = FNO2D(modes, modes, width, n_layers).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()
        optimizer.step()
        
        train_mse += mse.item()
        train_l2 += loss.item()
    
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_mse /= len(train_loader)
    train_l2/= ntrain
    test_l2 /= ntest
    
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2
    
    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)

