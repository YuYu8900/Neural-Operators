import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
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
args.in_dim = 2
args.out_dim = 1
args.h = 40
args.w = 40
args.h_down =args.w_down= 1
args.batch_size = 20
args.learning_rate = 0.001 
args.model='LSM_1D' 
args.d_model=64
args.num_basis = 12
args.num_token = 4 
args.patch_size='3 '
args.padding='8 '

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

data = np.load('/home/yy/no_survey/data/time_advection/train_IC2.npz')
x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
u0_train = u_train[:ntrain, 0, :]  # N x nx
x = np.repeat(x[0:1, :], ntrain, axis=0)  # N x nx
x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_train = u_train.transpose(0, 2, 1)[:ntrain,:,:]  # N x nx x nt
x_train = torch.from_numpy(x_train)
u_train = torch.from_numpy(u_train)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)

data = np.load('/home/yy/no_survey/data/time_advection/test_IC2.npz')
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
u0_test = u_test[:ntest, 0, :]  # N x nx
x = np.repeat(x[0:1, :], ntest, axis=0)  # N x nx
x_test = np.concatenate((u0_test[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
u_test = u_test.transpose(0, 2, 1)[:ntest,:,:]  # N x nx x nt
x_test = torch.from_numpy(x_test)
u_test = torch.from_numpy(u_test)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size, shuffle=False)



################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

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
        
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    train_loss[ep] = train_l2_step/ntrain/(T/step)
    test_loss[ep] = test_l2_step/ntest/(T/step)
    
    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)

# %%
""" Prediction """
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
