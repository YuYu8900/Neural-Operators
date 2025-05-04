import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.adam import Adam
from utils.params import get_args
from model_dict import get_model
import math
import os


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

TRAIN_PATH = '/burgers_data_512_51.mat'
# ntrain = 480
# ntest = 20

args.ntrain = 1000
args.ntest = 100
args.ntotal = 2000
args.in_dim = 1
args.out_dim = 1
args.h = 40
args.w = 40
args.h_down =args.w_down= 1
args.batch_size = 50
args.learning_rate = 0.001 
args.model='LSM_2D' 
args.d_model=64
args.num_basis = 12
args.num_token = 4 
args.patch_size='3,3 '
args.padding='8,8'

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
# s1 = 512

# batch_size = 20 
# learning_rate = 0.0005
# epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

# model_save_path = args.model_save_path
# model_save_name = args.model_save_name

T = 40
step = 1
step = 1

################################################################
# models
################################################################
model = get_model(args)

print(count_params(model))

################################################################
# load data and data normalization
################################################################
""" Read data """

data = np.load('/time_advection/train_IC2.npz')
x, t, u_train = data["x"], data["t"], data["u"] 
u_train = torch.tensor(u_train, dtype=torch.float) # N x nt x nx
# x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
x_train = u_train[:, 0:1, :]  # N x nx
x_train = x_train.reshape(ntrain,s1,1,1).repeat([1,1,40,1])
# x = np.repeat(x[0:1, :], ntrain, axis=0)  # N x nx
# x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_train = u_train.permute(0, 2, 1)  # N x nx x nt
# x_train = torch.from_numpy(x_train)
# u_train = torch.from_numpy(u_train)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)

data = np.load('/time_advection/test_IC2.npz')
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
u_test  = torch.tensor(u_test, dtype=torch.float)
x_test = u_test[:ntest, 0, :].reshape(ntest,s1 ,1)   # N x nx
x_test  = x_test.reshape(ntest,s1,1,1).repeat([1,1,40,1])
# x = np.repeat(x[0:1, :], ntest, axis=0)  # N x nx
# x_test = np.concatenate((u0_test[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_test = u_test[:ntest, :, :].permute(0, 2, 1)  # N x nx x nt
# x_test = torch.from_numpy(x_test)
# u_test = torch.from_numpy(u_test)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size, shuffle=False)



################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

device = torch.device('cuda')

# Remove weight_decay doesn't help.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        
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

            out = model(x)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_mse /= len(train_loader)
    train_l2/= ntrain
    test_l2 /= ntest
    
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2
    
    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)