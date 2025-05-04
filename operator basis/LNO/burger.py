import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

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
case = "Burger/" + t0 +'/'
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
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 +x2
        # x = torch.sin(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = torch.sin(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = torch.sin(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

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
sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h
ntrain = 1000
ntest = 100

batch_size_train = 20
batch_size_vali = 20

learning_rate = 0.005
epochs = 1000
step_size = 100
gamma = 0.5

modes = 16
width = 4

print(f'modes:{modes},width:{width}')
dataloader = MatReader('burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
grid_x_train = torch.tensor(np.linspace(0, 1, s), dtype=torch.float).reshape(s, 1)

x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]
grid_x_test = torch.tensor(np.linspace(0, 1, s), dtype=torch.float).reshape(s, 1)

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size_vali, shuffle=False)


# model
model = LNO1d(width,modes).cuda()


# ====================================
# Training 
# ====================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=True)

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    n_train=0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        t=grid_x_train.cuda()
        optimizer.zero_grad()
        out = model(x)   
        mse = F.mse_loss(out.view(batch_size_train, -1), y.view(batch_size_train, -1), reduction='mean')
        l2 = myloss(out.view(batch_size_train, -1), y.view(batch_size_train, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()
        n_train += 1

    scheduler.step()
    model.eval()
    vali_mse = 0.0
    vali_l2 = 0.0
    with torch.no_grad():
        n_vali=0
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            t=grid_x_test.cuda()
            out = model(x)
            mse=F.mse_loss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1), reduction='mean')
            vali_l2 += myloss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1)).item()
            vali_mse += mse.item()
            n_vali += 1

    train_mse /= n_train
    vali_mse /= n_vali
    train_l2 /= n_train
    vali_l2 /= n_vali
    train_error[ep,0] = train_l2
    vali_error[ep,0] = vali_l2
    train_loss[ep,0] = train_mse
    vali_loss[ep,0] = vali_mse
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train Loss: %.3e,Vali Loss: %.3e, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_mse, vali_mse,train_l2, vali_l2))
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
np.savetxt(save_results_to+'/vali_loss.txt', vali_loss)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'Wave_states')

# ====================================
# saving settings
# ====================================
pred = torch.zeros(y_test.shape)
index = 0
test_l2 = 0.0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=1, shuffle=False)

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        # t=grid_x_test.cuda()
        out = model(x).view(1,-1)
        pred[index]= out
        test_l2 += myloss(out, y).item()
        index = index + 1
test_l2/=index

scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={'test_err': test_l2,
                            'y_test': y_test.numpy(), 
                            'y_pred': pred.cpu().numpy()})  
    
    
print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")

