import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from utils.adam import Adam
import math
import os
torch.set_num_threads(5)


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

TRAIN_PATH = 'ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'ns_data_V100_N1000_T50_2.mat'

# TRAIN_PATH = os.path.join(args.data_path, './NavierStokes_V1e-5_N1200_T20.mat')
# TEST_PATH = os.path.join(args.data_path, './NavierStokes_V1e-5_N1200_T20.mat')
args.ntrain = 1000
args.ntest = 20
args.ntotal = 1200
args.in_dim = 10
args.out_dim = 1
args.h = 64
args.w = 64
args.h_down =args.w_down= 1
args.batch_size = 10
args.learning_rate = 0.0005 
args.model='LSM_3D' 
args.d_model=64
args.num_basis = 12
args.num_token = 4 
args.patch_size='5,5,3 '
args.padding='8,16,16'

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = args.T_in
T_out = args.T_out

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

sub = 1
S = 64 // sub
T_in = 10
T = 40

################################################################
# models
################################################################
model = get_model(args)
print(count_params(model))

################################################################
# load data and data normalization
################################################################

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

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).view(batch_size, S, S, T)

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

            out = model(x).view(batch_size, S, S, T)
            out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
# torch.save(model, path_model)


pred = torch.zeros(test_u.shape)
index = 0
test_e = torch.zeros( test_u.shape[0])
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
        test_e[index] = test_l2
        index = index + 1
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')