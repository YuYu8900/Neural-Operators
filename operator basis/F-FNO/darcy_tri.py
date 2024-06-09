"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utilities3 import *
from grid_2d import SpectralConv2d as FactorizedSpectralConv2d
from timeit import default_timer

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32, transform=True):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        if transform:
            self.scale = (1 / (in_channels * out_channels))
            self.weights1 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(
                self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None, transform=True):
        batchsize = u.shape[0]

        # Compute Fourier coefficients up to factor of e^(- something constant)
        if x_in is None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        if transform:
            factor1 = self.compl_mul2d(
                u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            factor2 = self.compl_mul2d(
                u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        else:
            factor1 = u_ft[:, :, :self.modes1, :self.modes2]
            factor2 = u_ft[:, :, -self.modes1:, :self.modes2]

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1,
                                 s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n_points, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        B = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1),
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1),
                          torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(device)

        # Shift the mesh coords into the right location on the unit square.
        if iphi is None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # x.shape == [B, N, 2]
        # K = <y, k_x>,  (batch, N, m1, m2)
        K = torch.outer(x[..., 0].view(-1), k_x1.view(-1)
                         ).reshape(B, N, N, m1, m2)
        # K2 = torch.outer(x[..., 1].view(-1), k_x2.view(-1)
        #                  ).reshape(B, N,N, m1, m2)
        # K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcmn,bmnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 = torch.cat((torch.arange(start=0, end=self.modes1, step=1),
                          torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1, 1).repeat(1, m2).to(device)
        k_x2 = torch.cat((torch.arange(start=0, end=self.modes2, step=1),
                          torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1, m2).repeat(m1, 1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:, :,:, 0].view(-1), k_x1.view(-1)
                         ).reshape(batchsize, N,N, m1, m2)
        # K2 = torch.outer(x[:, :, 1].view(-1), k_x2.view(-1)
        #                  ).reshape(batchsize, N, m1, m2)
        K = K1 #+ K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bmnxy->bcmn", u_ft, basis)
        Y = Y.real
        return Y


class FNOFactorizedPointCloud2D(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels,
                 n_layers=4, is_mesh=True, s1=40, s2=40, share_weight=False):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2
        self.n_layers = n_layers

        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(in_channels, self.width)

        self.convs = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        self.bs = nn.ModuleList([])

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(width, width, modes1, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        for i in range(self.n_layers + 1):
            if i == 0:
                conv = SpectralConv2d(
                    self.width, self.width, self.modes1, self.modes2, s1, s2, transform=False)
            elif i == self.n_layers:
                conv = SpectralConv2d(
                    self.width, self.width, self.modes1, self.modes2, s1, s2)
            else:
                conv = FactorizedSpectralConv2d(in_dim=width,
                                                out_dim=width,
                                                n_modes=modes1,
                                                forecast_ff=None,
                                                backcast_ff=None,
                                                fourier_weight=self.fourier_weight,
                                                factor=2,
                                                ff_weight_norm=True,
                                                n_ff_layers=2,
                                                layer_norm=False,
                                                use_fork=False,
                                                dropout=0.0,
                                                mode='full')
            self.convs.append(conv)

        self.bs.append(nn.Conv2d(2, self.width, 1))
        self.bs.append(nn.Conv2d(1, self.width, 1))

        for i in range(self.n_layers - 1):
            w = nn.Conv2d(self.width, self.width, 1)
            self.ws.append(w)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u.shape == [batch_size, n_points, 2] are the coords.
        # code.shape == [batch_size, 42] are the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in is None:
            x_in = u
        if self.is_mesh and x_out is None:
            x_out = u

        # grid is like the (x, y) coordinates of a unit square [0, 1]^2
        grid = self.get_grid([u.shape[0], self.s1, self.s2],
                             u.device).permute(0, 3, 1, 2)
        # grid.shape == [batch_size, 2, size_x, size_y] == [20, 2, 40, 40]
        # grid[:, 0, :, :] is the row index (y-coordinate)
        # grid[:, 1, :, :] is the column index (x-coordinate)

        # Projection to higher dimension
        u = self.fc0(u)
        u = u.permute(0, 3, 1, 2)
        # u.shape == [batch_size, hidden_size, n_points]

        uc1 = self.convs[0](u, x_in=x_in, iphi=iphi,
                            code=code, transform=False)  # [20, 32, 40, 40]
        uc3 = self.bs[0](grid)
        uc = uc1 + uc3

        # uc.shape == [20, 32, 40, 40]
        for i in range(1, self.n_layers):
            uc1 = rearrange(uc, 'b c h w -> b h w c')
            uc1 = self.convs[i](uc1)[0]
            uc1 = rearrange(uc1, 'b h w c -> b c h w')
            # uc2 = self.ws[i-1](uc)
            uc3 = self.bs[0](grid)
            uc = uc + uc1 + uc3

        L = self.n_layers
        u = self.convs[L](uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.bs[-1](x_out.permute(0, 3, 1, 2))
        u = u + u3

        u = u.permute(0, 2, 3, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class IPHI(nn.Module):
    def __init__(self, width=32):
        super(IPHI, self).__init__()

        """
        inverse phi: x -> xi
        """
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.fc_code = nn.Linear(42, self.width)
        self.fc_no_code = nn.Linear(3*self.width, 4*self.width)
        self.fc1 = nn.Linear(4*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 4*self.width)
        self.fc3 = nn.Linear(4*self.width, 4*self.width)
        self.fc4 = nn.Linear(4*self.width, 2)
        self.activation = torch.tanh
        self.center = torch.tensor([0.0001,0.0001], device="cuda").reshape(1,1,2)

        self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,self.width//4)


    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd


""" Model configurations """

PATH = '/home/fcx/yy/data/Darcy_Triangular_FNO.mat'
ntrain = 1900
ntest = 100

batch_size = 25
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.75


modes = 12
width = 32
n_layers = 4
input_dim = 1
output_dim = 1
share_weight = 'false'

r = 2
h = int(((101 - 1)/r) + 1) #51
s = h

# %%
""" Read data """
reader = MatReader(PATH)
x_train = reader.read_field('boundCoeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
# corrd_x
x_test = reader.read_field('boundCoeff')[-ntest:,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[-ntest:,::r,::r][:,:s,:s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = FNOFactorizedPointCloud2D(modes,modes,width,input_dim,output_dim,
                                  n_layers,'True',s,s,share_weight).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        
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

            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_mse /= len(train_loader)
    train_l2/= ntrain
    test_l2 /= ntest
    
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2
    
    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
    
# %%
""" Prediction """
pred = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(s, s)
        out = y_normalizer.decode(out)
        pred[index] = out

        test_l2 += myloss(out.reshape(1, s, s), y.reshape(1, s, s)).item()
        test_e[index] = test_l2
        print(index, test_l2)
        index = index + 1
        
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')