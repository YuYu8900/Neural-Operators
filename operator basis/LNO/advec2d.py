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
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================  

class PR2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(PR2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, dtype=torch.cfloat))
    
    def output_PR(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik->pqbixk",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2)))
        Hw=torch.einsum("bixk,pqbixk->pqbixk",weights_residue,term1)
        Pk=Hw  # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw; 
        output_residue1=torch.einsum("biox,oxikpq->bkox", alpha, Hw) 
        output_residue2=torch.einsum("biox,oxikpq->bkpq", alpha, Pk) 
        return output_residue1,output_residue2

    def forward(self, x):
        tx=T.cuda()
        ty=X.cuda()
        #Compute input poles and resudes by FFT
        dty=(ty[0,1]-ty[0,0]).item()  # location interval
        dtx=(tx[0,1]-tx[0,0]).item()  # time interval
        alpha = torch.fft.fft2(x, dim=[-2,-1])
        omega1=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # location frequency
        omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # time frequency
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=omega1.cuda()
        lambda2=omega2.cuda()
 
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)

        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
        x1 = torch.real(x1)    
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, ty.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bipz,biqx->bipqzx", torch.exp(term1),torch.exp(term2))
        x2=torch.einsum("kbpq,bipqzx->kizx", output_residue2,term3)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)
        return x1+x2

class LNO2d(nn.Module):
    def __init__(self, width,modes1,modes2):
        super(LNO2d, self).__init__()

        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.fc0 = nn.Linear(3, self.width) 

        self.conv0 = PR2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.conv1 = PR2d(self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.conv2 = PR2d(self.width, self.width, self.modes1, self.modes2)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.conv3 = PR2d(self.width, self.width, self.modes1, self.modes2)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 +x2
        x = torch.tanh(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x2 = self.w1(x)
        x = x1 +x2
        x = torch.tanh(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x2 = self.w2(x)
        x = x1 +x2
        x = torch.tanh(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x2 = self.w3(x)
        x = x1 +x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# ====================================
#  Define parameters and Load data
# ====================================
ntrain = 1000
ntest = 100
nt = 40
nx = 40
s = 40

batch_size_train = 20
batch_size_vali = 20

learning_rate = 0.002

epochs = 1000
step_size = 100
gamma = 0.5

modes1 = 4  
modes2 = 4   
width = 48

data = np.load('/time_advection/train_IC2.npz')
x, t, u_train = data["x"], data["t"], data["u"] 
X = torch.from_numpy(x[0:1,:].reshape(1,nx))
T= torch.from_numpy(t[:,0:1].reshape(1,nt))
u_train = torch.tensor(u_train, dtype=torch.float) # N x nt x nx
# x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
x_train = u_train[:, 0:1, :]  # N x nx
x_train = x_train.reshape(ntrain,nx,1,1).repeat([1,1,40,1])
# x = np.repeat(x[0:1, :], ntrain, axis=0)  # N x nx
# x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_train = u_train.permute(0, 2, 1)  # N x nx x nt
# x_train = torch.from_numpy(x_train)
# u_train = torch.from_numpy(u_train)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size_train, shuffle=True)

data = np.load('/time_advection/test_IC2.npz')
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
u_test  = torch.tensor(u_test, dtype=torch.float)
x_test = u_test[:ntest, 0, :].reshape(ntest,nx ,1)   # N x nx
x_test  = x_test.reshape(ntest,nx,1,1).repeat([1,1,40,1])
# x = np.repeat(x[0:1, :], ntest, axis=0)  # N x nx
# x_test = np.concatenate((u0_test[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_test = u_test[:ntest, :, :].permute(0, 2, 1)  # N x nx x nt
# x_test = torch.from_numpy(x_test)
# u_test = torch.from_numpy(u_test)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size_vali, shuffle=False)


model = LNO2d(width,modes1, modes2).cuda()

# ====================================
# Training 
# ====================================
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=False)

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

        optimizer.zero_grad()
        out = model(x)   
        mse = F.mse_loss(out.view(batch_size_train, -1), y.view(batch_size_train, -1), reduction='mean')
        l2 = myloss(out.view(-1,x_train.shape[1],x_train.shape[2]), y)
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
            out = model(x)
            mse=F.mse_loss(out.view(-1,x_test.shape[1],x_test.shape[2]), y, reduction='mean')
            vali_l2 += myloss(out.view(-1,x_test.shape[1],x_test.shape[2]), y).item()
            vali_mse += mse.item()
            n_vali += 1

    train_mse /= ntrain
    vali_mse /= ntest
    train_l2 /= ntrain
    vali_l2 /= ntest
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
t0 = time.strftime('%y%m%d_%H_%M_%S')
   
current_directory = os.getcwd()
case = "Advection/" + t0 +'/'


results_dir = "/train/result/" + case
save_results_to = current_directory + results_dir
print(f"save_results_to:{save_results_to}")
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
# testing
# ====================================
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=1, shuffle=False)
pred_u = torch.zeros(ntest,u_test.shape[1],u_test.shape[2])
index = 0
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        test_l2 += myloss(out.view(-1,x_test.shape[1],x_test.shape[2]), y).item()
        pred_u[index,:,:] = out.view(-1,x_test.shape[1],x_test.shape[2])
        index = index + 1
test_l2 /= index
scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={ 'test_err': test_l2,
                            'y_test': u_test.cup.numpy(), 
                            'y_pred': pred_u.cpu().numpy()})  
    
    
print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")
