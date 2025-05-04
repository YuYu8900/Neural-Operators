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
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================  

class PR3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(PR3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_pole3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes3, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, self.modes3, dtype=torch.cfloat))

    def output_PR(self, lambda1, lambda2, lambda3, alpha, weights_pole1, weights_pole2, weights_pole3, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],weights_residue.shape[4],lambda1.shape[0], lambda2.shape[0], lambda2.shape[3], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik,rbio->pqrbixko",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2),torch.sub(lambda3,weights_pole3)))
        Hw=torch.einsum("bixko,pqrbixko->pqrbixko",weights_residue,term1)
        output_residue1=torch.einsum("bioxs,oxsikpqr->bkoxs", alpha, Hw) 
        output_residue2=torch.einsum("bioxs,oxsikpqr->bkpqr", alpha, -Hw) 
        return output_residue1,output_residue2
    

    def forward(self, x):
        tt=T.cuda()
        tx=X.cuda()
        ty=Y.cuda()
        #Compute input poles and resudes by FFT
        dty=(ty[0,1]-ty[0,0]).item()  # location interval
        dtx=(tx[0,1]-tx[0,0]).item()  # location interval
        dtt=(tt[0,1]-tt[0,0]).item()  # time interval
        alpha = torch.fft.fftn(x, dim=[-3,-2,-1])
        omega1=torch.fft.fftfreq(tt.shape[1], dtt)*2*np.pi*1j   # time frequency
        omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # location frequency
        omega3=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # location frequency
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega3=omega3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=omega1.cuda()
        lambda2=omega2.cuda()    
        lambda3=omega3.cuda()

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda2, lambda3,lambda1, alpha, self.weights_pole1, self.weights_pole2, self.weights_pole3, self.weights_residue)
 
      
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifftn(output_residue1, s=(x.size(-3),x.size(-2), x.size(-1)))
        x1 = torch.real(x1)
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, tt.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bim,ky->bimy", self.weights_pole3, ty.type(torch.complex64).reshape(1,-1))
        term4=torch.einsum("bipz,biqx,bimy->bipqmzxy", torch.exp(term2),torch.exp(term3),torch.exp(term1))
        x2=torch.einsum("kbpqm,bipqmzxy->kizxy", output_residue2,term4)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)/x.size(-3)
        return x1+x2

class LNO3d(nn.Module):
    def __init__(self, width,modes1,modes2,modes3):
        super(LNO3d, self).__init__()

        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.fc0 = nn.Linear(13, self.width) 

        self.conv0 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.conv1 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        # self.conv2 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.w2 = nn.Conv3d(self.width, self.width, 1)
        # self.conv3 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm3d(self.width)

        self.fc1 = nn.Linear(self.width,64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self,x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 +x2
        x = F.relu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x2 = self.w1(x)
        x = x1 +x2
        # x = F.relu(x)

        # x1 = self.norm(self.conv2(self.norm(x)))
        # x2 = self.w2(x)
        # x = x1 +x2
        # x = F.relu(x)

        # x1 = self.norm(self.conv3(self.norm(x)))
        # x2 = self.w3(x)
        # x = x1 +x2

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

# ====================================
#  Define parameters and Load data
# ====================================
TRAIN_PATH = 'ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'ns_data_V100_N1000_T50_2.mat'
ntrain = 1000
ntest = 200


batch_size = 10
epochs = 500
learning_rate = 0.002
step_size = 100
gamma = 0.5

sub = 1
S = 64
T_in = 10
T = 10
T1 = T
step = 1

modes1 = 4
modes2 = 4
modes3 = 4
width = 8

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])


a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)

train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

x = np.linspace(0, 1, S)
y = np.linspace(0, 1, S)
z = np.linspace(0, 1, T)
tt, xx, yy = np.meshgrid(z, x, y, indexing='ij')

T=torch.linspace(0,10,T).reshape(1,T)
X=torch.linspace(0,1,steps=S).reshape(1,S)[:,:S]
Y=torch.linspace(0,1,steps=S).reshape(1,S)[:,:S]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

device = torch.device('cuda') 
# model
model = LNO3d(width,modes1, modes2,modes3).cuda()


# ====================================
# Training 
# ====================================
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=False)
train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).view(batch_size, S, S, T1)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).view(batch_size, S, S, T1)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
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
case = "NS3d/" + t0 +'/'


results_dir = "/train/result/" + case
save_results_to = current_directory + results_dir
print(f"save_results_to:{save_results_to}")
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_error.txt', train_l2)
np.savetxt(save_results_to+'/vali_error.txt', test_l2)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'Wave_states')

# ====================================
# Testing 
# ====================================
pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x)
        out = y_normalizer.decode(out)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1
scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={ 'test_err': test_l2,
                            'T': T.numpy(),
                            'X': X.numpy(),
                            'Y': Y.numpy(),
                            'y_test': test_a.numpy(), 
                            'y_pred': pred.cpu().numpy(),
                            'Train_time':elapsed})  
    
    
print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")

