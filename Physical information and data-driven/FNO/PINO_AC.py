# from matplotlib.pyplot import flag
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
from utilities3 import *
from timeit import default_timer
from losses import LpLoss,AC_loss
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


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
        self.padding = 1  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)  # input channel is 3: (a(x, y), x, y)
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

""" Model configurations """

TRAIN_PATH = 'Allen_Cahn_pde_65_65_1000.mat'


ntrain = 600
ntest = 20

batch_size = 60
learning_rate = 0.0001

epochs = 500
step_size = 10
gamma = 0.5

modes = 16 
width = 32

n_layers = 4

r = 1
h = int(((65 - 1)/r) + 1)
s = h
s2 = 33
T_in = 10
T = 10
step = 1
S = 65

data_weight = 1
pde_weight = 0

# %%
""" Read data """
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('sol').permute(3,0,1,2)
# y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

train_a = x_train[:ntrain,:,:,:T_in]
train_u = x_train[:ntrain,:,:,T_in:T+T_in]

test_a = x_train[-ntest:,:,:,:T_in]
test_u = x_train[-ntest:,:,:,T_in:T+T_in]


train_a = train_a.reshape(ntrain,s,s,T_in)
test_a = test_a.reshape(ntest,s,s,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

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
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_dl = 0
    train_pl = 0
    for xx, yy in train_loader:    
        data_loss = 0
        xx = xx.to(device)
        yy = yy.to(device)  
        xx = xx[:,::2,::2,:]
        yy = yy[:,::2,::2,:]   
        loss_physics = 0  
        for t in range(0, T, step):
            y = yy[..., t:t + step]  # t:t+step, retains the third dimension, butt only t don't,
            # y = yy[..., t:t + step]
            im = model(xx)  
            # im2 = model(xx2)  
            data_loss += F.mse_loss(im, y, reduction = 'sum') 
            # data_loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)
            
            if t == 0: 
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
    
            xx = torch.cat((xx[..., step:], im), dim=-1)
            # xx2 = torch.cat((xx2[..., step:], im2), dim=-1)
            
            if pde_weight == 0:
                loss_physics = torch.zeros(1, device=device)
            else:
                for kk in range(im.shape[0]):
                    y_pred = im[kk,:,:,:].squeeze(-1)
                    y_pf = y_pred.reshape(s2,s2)
                    x_pf = xx[kk,:,:,-2].reshape(s2,s2)
                    y_dash = y[kk,:,:,:].squeeze(-1)
                    #bound condition 1
                    tp_u = y_pred[0,:][:,None]
                    tp_usol = y_dash[0,:][:,None] 
                    bt_u = y_pred[-1,:][:,None]
                    bt_usol = y_dash[-1,:][:,None] 
                    
                    lt_u =  y_pred[:,0][:,None] #L2
                    lt_usol = y_dash[:,0][:,None] #L2
                    rt_u = y_pred[:,-1][:,None]
                    rt_usol = y_dash[:,-1][:,None]
                    
                    all_u_train = torch.vstack([tp_u,bt_u,lt_u,rt_u]) # X_u_train [200,2] (800 = 200(L1)+200(L2)+200(L3)+200(L4))
                    all_u_sol = torch.vstack([tp_usol,bt_usol,lt_usol,rt_usol])   #corresponding u [800x1]
                    loss_physics += AC_loss(all_u_train,all_u_sol,x_pf,y_pf)
        
        loss =  data_weight*data_loss + pde_weight*loss_physics

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_l2 += loss.item()
        train_dl += data_loss.item()
        train_pl += loss_physics.item()
        
    scheduler.step()
    model.eval()
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            xx2 = xx[:,::2,::2,:]
            yy2 = yy[:,::2,::2,:]
        
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                y2 = yy2[..., t:t + step]
                im2 = model(xx2)
                loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)
        
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
        
                xx = torch.cat((xx[..., step:], im), dim=-1)
                
            test_l2_step += loss.item()
            test_l2_full += (torch.norm(pred-yy, p=2)/torch.norm(yy, p=2)).item()
            
            
    train_l2 /= (ntrain*T)
    train_dl /= (ntrain*T)
    train_pl /= (ntrain*T)
    
    test_l2_step /= (ntest*T)
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2_step
    
    t2 = default_timer()
    print('Epoch %d - Time %0.4f - Train %0.4f - PDE %0.4f - data %0.4f - Test %0.4f' 
          % (ep, t2-t1, train_l2, train_pl, train_dl, test_l2_step))    
    
    
# %%
""" Prediction """
print('sub=1')
pred0 = torch.zeros(test_u.shape)
index = 0
test_e = torch.zeros(test_u.shape)        
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

with torch.no_grad():
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)
            # loss += (torch.mean((im-y)**2)/torch.mean(y**2))
            # loss += (torch.linalg.norm(im-y,dim=(1,2,3))/torch.linalg.norm(y,dim=(1,2,3)))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        pred0[index] = pred
        test_l2_step += loss.item()
        test_l2_full += (torch.norm(pred-yy, p=2)/torch.norm(yy, p=2)).item()
        # test_l2_full += (torch.mean((pred-yy)**2)/torch.mean(yy**2)).item()
        # test_l2_full += (torch.linalg.norm(pred-yy,dim=(1,2,3))/torch.linalg.norm(yy,dim=(1,2,3))).item()
        test_e[index] =  test_l2_step/ ntest/ (T/step)
        
        print(index, test_l2_step/ ntest/ (T/step), test_l2_full/ ntest)
        index = index + 1
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')
print('STD Testing Error:', 100*torch.std(test_e).numpy() , '%')  

print('--------------------------------------------------------------------------------')
print('sub=2')
pred0 = torch.zeros(test_u[:,::2,::2,:].shape)
index = 0
test_e = torch.zeros(test_u[:,::2,::2,:].shape)        
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a[:,::2,::2,:], test_u[:,::2,::2,:]), batch_size=1, shuffle=False)

with torch.no_grad():
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)
            # loss += (torch.mean((im-y)**2)/torch.mean(y**2))
            # loss += (torch.linalg.norm(im-y,dim=(1,2,3))/torch.linalg.norm(y,dim=(1,2,3)))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        pred0[index] = pred
        test_l2_step += loss.item()
        test_l2_full += (torch.norm(pred-yy, p=2)/torch.norm(yy, p=2)).item()
        # test_l2_full += (torch.mean((pred-yy)**2)/torch.mean(yy**2)).item()
        # test_l2_full += (torch.linalg.norm(pred-yy,dim=(1,2,3))/torch.linalg.norm(yy,dim=(1,2,3))).item()
        test_e[index] =  test_l2_step/ ntest/ (T/step)
        
        print(index, test_l2_step/ ntest/ (T/step), test_l2_full/ ntest)
        index = index + 1
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')
print('STD Testing Error:', 100*torch.std(test_e).numpy() , '%')  
 