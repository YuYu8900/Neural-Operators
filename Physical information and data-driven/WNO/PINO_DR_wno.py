"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from losses import LpLoss,DR_loss
from torch.autograd import Variable

torch.manual_seed(0)
np.random.seed(0)


class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.dwt_ = DWT(J=self.level, mode='symmetric', wave='db6').to(dummy.device)
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=self.level, mode='symmetric', wave='db6').to(device)
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        x_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        x_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave='db6').to(device)
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
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 1 # pad the domain when required
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, self.dummy_data)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 192)
        self.fc2 = nn.Linear(192, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding,0,(self.padding)])

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

        x = x[..., :-self.padding, :-(self.padding)] 
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

################################################################
#  configurations
################################################################
ntrain = 800
ntest = 200

nx = 100
nt = 100
sub = 1
sub_t = 1
s = nt

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes = 16
width = 32
n_layers=4

#0，1，100
#5，1，1
data_weight = 1
f_weight = 0
ic_weight = 0

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('/home/yy/no_survey/data/Diffusion_reaction_1200_100_100_0.2.mat')
x_data = dataloader.read_field('u')
y_data = dataloader.read_field('UU')
# x = dataloader.read_field('x')
x_train = x_data[:ntrain,:].reshape(ntrain, nx, 1,1).repeat([1, 1, s,1])
y_train = y_data[:ntrain,:,:]

x_test = x_data[-ntest:,:].reshape(ntest, nx, 1,1).repeat([1, 1, s,1])
y_test = y_data[-ntest:,:,:]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
# model
level = 4
model = WNO2d(width, level, x_train.permute(0,3,1,2)).to(device)
print(count_params(model))

################################################################
# training and evaluation
################################################################
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=True)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_pino = 0.0
    data_l2 = 0.0
    train_loss = 0.0
    # dataloader_iterator = iter(train_loader2)
    for x, y in train_loader: 
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)
        # mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')

        data_loss = myloss(out, y)
        if ic_weight != 0 or f_weight != 0:
            # out = model(x)
            loss_ic, loss_f = DR_loss(out, x[:, :, 0, 0])
            # loss_ic, loss_f = burgers_loss(out, x[:, 0, :, 0], v)
            train_pino += loss_f.item()
        else:
            loss_ic, loss_f = 0, 0

        total_loss = loss_ic * ic_weight + loss_f * f_weight + data_loss * data_weight
        
        total_loss.backward() # use the l2 relative loss

        optimizer.step()
        # train_mse += mse.item()
        data_l2 += data_loss.item()
        train_loss += total_loss.item()
        
   
    data_l2 /= len(train_loader)
    train_pino /= len(train_loader)
    train_loss /= len(train_loader)
    scheduler.step()
    
    model.eval()
    
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(out, y)
            # test_loss.append(data_loss.item())

    # train_mse /= len(train_loader)
    # train_l2 /= ntrain
    test_l2 /= len(test_loader)

    t2 = default_timer()
    print(f'Epoch: {ep},T:{t2-t1:.5f},train loss: {train_loss:.5f},data loss:{data_l2:.5f}, f_loss: {train_pino:.5f},test loss:{test_l2:.5f} ')

# torch.save(model, 'model/ns_fourier_burgers')
# pred = torch.zeros(y_test.shape)
# index = 0
print('--------------------------------------------------------------------------------')
print('Predict')
index = 0
test_e = torch.zeros(y_test.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x)

        test_l2 = myloss(out, y)
        test_e[index] = test_l2
        # print(index, test_l2)
        index = index + 1

print('Mean Error:', 100*torch.mean(test_e))
# scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})