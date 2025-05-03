from matplotlib.pyplot import flag
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utilities3 import *
from timeit import default_timer
from losses import LpLoss,AC_loss
from timeit import default_timer
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

torch.manual_seed(0)
np.random.seed(0)

# %%
""" Def: 2d Wavelet layer """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.dwt_ = DWT(J=self.level, mode='symmetric', wave='db4').to(dummy.device)
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        # self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        # self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        #Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=self.level, mode='symmetric', wave='db4').to(device)
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        # x_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        # x_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave='db4').to(device)
        x = idwt((out_ft, x_coeff))
        return x

""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO2d, self).__init__()

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

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width) 
        # input channel is 12: the solution of the previous 10 timesteps + 
        # 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding]) 

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] 
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
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

batch_size = 20
learning_rate = 0.0001

epochs = 500
step_size = 10
gamma = 0.5

level = 4 
width = 32

n_layers = 4

r = 1
h = int(((65 - 1)/r) + 1)
s = h
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
model = WNO2d(width, level, train_a.permute(0,3,1,2)).to(device)
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
        loss_physics = 0  
        for t in range(0, T, step):
            y = yy[..., t:t + step]  # t:t+step, retains the third dimension, butt only t don't,
            im = model(xx)  
            data_loss += F.mse_loss(im, y, reduction = 'sum') 
            # data_loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)
            
            if t == 0: 
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
    
            xx = torch.cat((xx[..., step:], im), dim=-1)
            
            if pde_weight == 0:
                loss_physics = torch.zeros(1, device=device)
            else:
                for kk in range(im.shape[0]):
                    y_pred = im[kk,:,:,:].squeeze(-1)
                    y_pf = y_pred.reshape(s,s)
                    x_pf = xx[kk,:,:,-2].reshape(s,s)
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
        
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
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
# model = torch.load('model/ns_wno_allencan_p_3.5mse1')
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


 