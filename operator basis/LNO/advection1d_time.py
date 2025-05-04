import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import time

# ====================================
# saving settings
# ====================================
t0 = time.strftime('%y%m%d_%H_%M_%S')
save_index = 1   
current_directory = os.getcwd()
case = "Advection/" + t0 +'/'
folder_index = str(save_index)

results_dir = "/train/result/" + case
save_results_to = current_directory + results_dir
print(f"save_results_to:{save_results_to}")

# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================    
class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()

        self.modes1 = modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
   
    def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.sub(lambda1,weights_pole))
        Hw=weights_residue*term1
        Pk=-Hw  # for ode, Pk equals to negative Hw
        output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2=torch.einsum("bix,xiok->bok", alpha, Pk) 
        return output_residue1,output_residue2    
    

    def forward(self, x):
        t=grid_x_train.cuda()
        #Compute input poles and resudes by FFT
        dt=(t[1]-t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.cuda()
        start=time.time()

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)

        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1) 
        x2=torch.einsum("bix,ioxz->boz", output_residue2,term2)
        x2=torch.real(x2)
        x2=x2/x.size(-1)
        return x1+x2

class LNO1d(nn.Module):
    def __init__(self, width,modes):
        super(LNO1d, self).__init__()

        self.width = width
        self.modes1 = modes
        self.fc0 = nn.Linear(2, self.width) 

        self.conv0 = PR(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.conv1 = PR(self.width, self.width, self.modes1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.conv2 = PR(self.width, self.width, self.modes1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.conv3 = PR(self.width, self.width, self.modes1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 +x2
        x = torch.tanh(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = torch.tanh(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x =  torch.tanh(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

# ====================================
#  Define parameters and Load data
# ==================================== 
ntrain = 1000
ntest = 100
s = 40

batch_size_train = 20
batch_size_vali = 20

learning_rate = 0.005
epochs = 1000
step_size = 100
gamma = 0.5

modes = 16
width = 4


T = 40
step = 1


print(f'modes:{modes},width:{width}')
data = np.load('/time_advection/train_IC2.npz')
x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
u0_train = u_train[:ntrain, 0, :]
grid_x_train =torch.from_numpy(x[0:1, :].reshape(s, 1)) 
x = np.repeat(x[0:1, :], ntrain, axis=0)  # N x nx
x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_train = u_train.transpose(0, 2, 1)[:ntrain,:,:]  # N x nx x nt
x_train = torch.from_numpy(x_train)
u_train = torch.from_numpy(u_train)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size_train, shuffle=True)

data = np.load('/time_advection/test_IC2.npz')
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
u0_test = u_test[:ntest, 0, :]  # N x nx
grid_x_test = torch.from_numpy(x[0:1, :].reshape(s, 1))
x = np.repeat(x[0:1, :], ntest, axis=0)  # N x nx
x_test = np.concatenate((u0_test[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2 
u_test = u_test.transpose(0, 2, 1)  # N x nx x nt
x_test = torch.from_numpy(x_test)
u_test = torch.from_numpy(u_test[:ntest,:,:])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size_vali, shuffle=False)

# model
model = LNO1d(width,modes).cuda()


# ====================================
# Training 
# ====================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=True)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        t=grid_x_train.to(device)
        
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size_train, -1), y.reshape(batch_size_train, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size_train, -1), yy.reshape(batch_size_train, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    test_mse = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            t=grid_x_test.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size_vali, -1), y.reshape(batch_size_vali, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size_vali, -1), yy.reshape(batch_size_vali, -1)).item()
            test_mse += F.mse_loss(pred.reshape(batch_size_vali, -1), yy.reshape(batch_size_vali, -1), reduction='mean').item()

    train_loss[ep] = train_l2_step/ntrain/(T/step)
    test_loss[ep] = test_l2_step/ntest/(T/step)
    test_mse /= len(test_loader)
    
    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, test_mse, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)
elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")

# ====================================
# saving settings
# ====================================
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_loss.txt', train_loss)
np.savetxt(save_results_to+'/test_loss.txt', test_loss)  
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'Wave_states')

# ====================================
# saving settings
# ====================================
pred0 = torch.zeros(u_test.shape)
index = 0     
test_e = torch.zeros(u_test.shape)   
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=1, shuffle=False)

with torch.no_grad():
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        mse = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(1, y.size()[-3], y.size()[-2]), y.reshape(1, y.size()[-3], y.size()[-2]))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        pred0[index] = pred
        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        mse += F.mse_loss(pred.reshape(1, -1), yy.reshape(1, -1), reduction='mean')
        test_e[index] = test_l2_step
        
        print(index, test_l2_step / (T/step), test_l2_full / (T/step), mse.cpu().numpy())
        index = index + 1

print('Mean Testing Error:', 100*torch.mean(test_e).numpy() /(T/step), '%')

scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={'y_test': u_test.numpy(), 
                            'y_pred': pred0.cpu().numpy()})  
    
    

