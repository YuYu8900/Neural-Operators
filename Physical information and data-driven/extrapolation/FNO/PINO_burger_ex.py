"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from losses import LpLoss,PINO_loss,burgers_loss
from torch.autograd import Variable

# torch.manual_seed(0)
# np.random.seed(0)


################################################################
#  1d fourier layer
################################################################
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

        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # x = x[..., :-self.padding, :-self.padding]
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

def add_gaussian_noise(data):
    noisy_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        a = data[i]
        norm_inf = np.max(np.abs(a))  # 计算矩阵的无穷范数
        noise = np.random.normal(0, 1, a.shape)  # 从标准正态分布生成随机数
        noisy_data[i] = data[i] + 0.1 * norm_inf * noise  # 应用噪声公式
    return noisy_data


################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 100

nx = 101
nt = 100
sub = 1
sub_t = 1
s = nt+1


batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes = 12
width = 32
n_layers=4

#0，1，10
#5，1，1
data_weight = 0
f_weight = 1
ic_weight = 10

################################################################
# read data
################################################################

# Data is of the shape (number of samples, grid size)
train_data = np.load('ex_data/burgers/burgers_train_ls_1.0_101_101.npz')
x_train_data =   torch.tensor(train_data["X_train0"], dtype=torch.float)[:,::sub]
# x_train1 =   torch.tensor(train_data["X_train1"], dtype=torch.float)[:,::sub]
y_train_data =  torch.tensor(train_data["y_train"], dtype=torch.float).reshape(1000,s,s)
v = 0.1

x_train = x_train_data[:ntrain,:].reshape(ntrain, nx, 1,1).repeat([1, 1, s,1])
y_train = y_train_data[:ntrain,:,:]


test_data = np.load('ex_data/burgers/burgers_test_ls_1.0_101_101.npz')
x_test_data = torch.tensor(test_data["X_test0"], dtype=torch.float)[:,::sub] #n*nx*nt
y_test_data =  torch.tensor(test_data["y_test"], dtype=torch.float).reshape(100,s,s)
x_test = x_test_data[:ntest,:].reshape(ntest, nx, 1,1).repeat([1, 1, s,1])
y_test = y_test_data[:ntest,:,:]

test_data1 = np.load('ex_data/burgers/burgers_test_ls_0.6_101_101.npz')
x_test_data1 =   torch.tensor(test_data1["X_test0"], dtype=torch.float)[:,::sub]
y_test_data1 =  torch.tensor(test_data1["y_test"], dtype=torch.float).reshape(100,s,s)
x_test1 = x_test_data1[:ntest,:].reshape(ntest, nx, 1,1).repeat([1, 1, s,1])
y_test1 = y_test_data1[:ntest,:,:]
# gridx = torch.tensor(np.linspace(0, 1, nx), dtype=torch.float)
# gridt = torch.tensor(np.linspace(0, 1, nt + 1)[1:], dtype=torch.float)
# gridx =  Variable(gridx.reshape(1, 1, nx).repeat([ntrain, nt, 1]),requires_grad=True)
# gridt =  Variable(gridt.reshape(1, nt, 1).repeat([ntrain, 1, nx]),requires_grad=True)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
test_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, y_test1), batch_size=batch_size, shuffle=False)
# model
model = FNO2D(modes,modes, width,n_layers).cuda()
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
        # x, y = x[:,::4,::4],y[:,::4,::4]

        optimizer.zero_grad()
        out = model(x)
        # mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')

        data_loss = myloss(out, y)
        if ic_weight != 0 or f_weight != 0:
            # out = model(x)
            loss_ic, loss_f = PINO_loss(out, x[:, :, 0, 0], v)
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
            # x, y = x[:,::4,::4],y[:,::4,::4]

            out = model(x)
            test_l2 += myloss(out, y)
            # test_loss.append(data_loss.item())

    # train_mse /= len(train_loader)
    # train_l2 /= ntrain
    test_l2 /= len(test_loader)

    ex_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader1:
            x, y = x.cuda(), y.cuda()
            # x, y = x[:,::4,::4],y[:,::4,::4]

            out = model(x)
            ex_l2 += myloss(out, y)
            # test_loss.append(data_loss.item())

    # train_mse /= len(train_loader)
    # train_l2 /= ntrain
    ex_l2 /= len(test_loader1)

    t2 = default_timer()
    print(f'Epoch: {ep},T:{t2-t1:.5f},train loss: {train_loss:.5f},data loss:{data_l2:.5f}, f_loss: {train_pino:.5f},test loss:{test_l2:.5f}, Ex loss:{ex_l2:.5f} ')

# torch.save(model, 'model/ns_fourier_burgers')
# pred = torch.zeros(y_test.shape)
# index = 0
