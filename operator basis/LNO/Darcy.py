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
        tx=X.cuda()
        ty=Y.cuda()
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
TRAIN_PATH = 'Darcy_421/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'Darcy_421/piececonst_r421_N1024_smooth2.mat'
ntrain = 1000
ntest = 100

r = 5
h = int(((421 - 1)/r) + 1)
s = h

batch_size_train = 20
batch_size_vali = 20

learning_rate = 0.005

epochs = 1000
step_size = 100
gamma = 0.5

modes1 = 4  
modes2 = 4   
width = 16

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

X = torch.tensor(np.linspace(0, 1, s), dtype=torch.float).reshape(1, s)
Y = torch.tensor(np.linspace(0, 1, h), dtype=torch.float).reshape(1, h)

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size_vali, shuffle=False)


model = LNO2d(width,modes1, modes2).cuda()

# ====================================
# Training 
# ====================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=False)

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    n_train=0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size_train, s, s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

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
            out = model(x).reshape(batch_size_vali, s, s)
            out = y_normalizer.decode(out)
            mse=F.mse_loss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1),reduction='mean')
            vali_l2 += myloss(out.view(batch_size_vali, -1), y.view(batch_size_vali, -1)).item()
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
save_index = 1   
current_directory = os.getcwd()
case = "Darcy/" + t0 +'/'
folder_index = str(save_index)

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
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
pred_u = torch.zeros(ntest,y_test.shape[1],y_test.shape[2])
index = 0
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x).reshape(1, s, s)
        out = y_normalizer.decode(out)
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        pred_u[index,:,:] = out.view(-1,x_test.shape[1],x_test.shape[2])
        index = index + 1
test_l2 /= index
scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={ 'test_err': test_l2,
                            'y_test': y_test.cpu().numpy(), 
                            'y_pred': pred_u.cpu().numpy()})  
    
    
print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")
