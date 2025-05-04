# from IPython import get_ipython
# get_ipython().magic('reset -sf')
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
# from Training_wno_weights import *
from WNO_encode import *
from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)
import time
seed = 2022
torch.manual_seed(seed)
np.random.seed(seed)
print(f'seed:{seed}')

# device= torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
# %%
""" Model configurations """

TRAIN_PATH = 'ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'ns_data_V100_N1000_T50_2.mat'

ntrain = 1000
ntest = 200

sub = 1
S = 64

num_epochs = 1
num_epochs2 = 300#300
# batch_size = 80
learning_rate = 5e-4


step_size = 20
gamma = 0.75

level = 4# The automation of the mode size will made and shared upon acceptance and final submission
width = 28

batch_size = 5

T_in = 9
# T = 10
T_out = 10
step =1

""" Read data """
reader = MatReader(TRAIN_PATH)
x_train_enc1 = reader.read_field('u')[:ntrain,::sub,::sub,0:T_in]
x_train_enc2 = reader.read_field('u')[:ntrain,::sub,::sub,1:T_in+1]
x_train_out1 = reader.read_field('u')[:ntrain,::sub,::sub,1:T_in+T_out+1]

reader = MatReader(TEST_PATH)
test_enc1 = reader.read_field('u')[-ntest:,::sub,::sub,0:T_in]
test_enc2 = reader.read_field('u')[-ntest:,::sub,::sub,1:T_in+1]
test_out1 = reader.read_field('u')[-ntest:,::sub,::sub,T_in+1:T_in+T_out+1]

# test_out2  = x_test[:,:,:,T_in+1:T_out+1]
# x_test_enc = time_windowing(x_test,T).permute(0,1,3,4,2).to(device)
# train_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_enc0, x_train_out0), batch_size=batch_size, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_enc1, x_train_enc2, x_train_out1), batch_size=batch_size, shuffle=True)
# test_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_enc0, test_out0), batch_size=batch_size, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_enc1, test_enc2, test_out1), batch_size=batch_size, shuffle=True)
test_loader_pred2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_enc1, test_enc2, test_out1), batch_size=1, shuffle=False)
#%%
# model = WNO2d(width, level, x_train_enc0[0:1,:,:,:].permute(0,3,1,2)).to(device)
# print(count_params(model))

# """ Training and testing """
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# n0_ws,n0_bs,n1_ws,n1_bs,n2_ws,n2_bs = training_model(model, optimizer,scheduler,num_epochs,train_loader1,test_loader1,batch_size,ntrain,ntest,T_in,T_out,step)
#%%
t0 = time.strftime('%y%m%d_%H_%M_%S')
model_path = 'NS_'+'width'+str(width)+'_level'+str(level)+'_epoch'+str(num_epochs2)+'_'+t0
print(model_path)

""" The model definition """
model2 = WNO2dtransformer(width, level, x_train_enc1[0:1,:,:,0:T_in].permute(0,3,1,2)).to(device)

# model2.load_state_dict(checkpoint['model'],strict=False)
print(count_params(model2))
""" Training and testing """
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=False)
for ep in range(num_epochs2):
    model2.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for (dat1, dat2, dat_out) in train_loader2:
        loss = 0
        xx = dat1.to(device)
        yy = dat2.to(device) 
        zz = dat_out.to(device) 
        loss_physics = 0  
        for t in range(0, T_out, step):
            # x = xx[:,:,:,t:t+T_in]
            # y = yy[:,:,:,t:t+T_in]
            z = zz[:,:,:,t+T_in:t+T_in+step]
            im =  model2(xx,yy) 
            loss += myloss(im.reshape(batch_size, -1), z.reshape(batch_size, -1)) 
            if t == 0:
               pred = im
            else:
               pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., 1:], yy[:,:,:,-1:]), dim=-1)
            yy = torch.cat((yy[..., 1:], im), dim=-1)
            
        train_l2_step += loss.item()
        l2_full = loss
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
    test_l2_step = 0
    test_l2_full = 0

    with torch.no_grad():
        for xx, yy, zz in test_loader2:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            zz = zz.to(device)
            for t in range(0, T_out, step):
                z = zz[:,:,:,t:t+step]
                im = model2(xx,yy)
                loss += myloss(im.reshape(batch_size, -1), z.reshape(batch_size, -1))
                if t == 0:
                   pred = im
                else:
                   pred = torch.cat((pred, im), -1)
                
                xx = torch.cat((xx[..., 1:], yy[:,:,:,-1:]), dim=-1)
                yy = torch.cat((yy[..., 1:], im), dim=-1)
                
            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), zz.reshape(batch_size, -1)).item()
            
    train_l2_step /= (ntrain*T_out)
    train_l2_full /= (ntrain*T_out)
    test_l2_step /= (ntest*T_out)
    t2 = default_timer()
    scheduler.step()
    print('Epoch %d - Time %0.4f - Train_l2_step %0.4f - data_l2 %0.6f - Test %0.6f' 
          % (ep, t2-t1, train_l2_step, train_l2_full, test_l2_step)) 
    checkpoint = {'model':model2.state_dict(),'optimizer':optimizer.state_dict()}
    torch.save(checkpoint, os.path.join('NS_2d/model/{}.pt'.format(model_path))) 
   
   
# %%
""" Prediction """
pred0 = torch.zeros(test_out1.shape)
index = 0
test_e = []       
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
t1 = default_timer()
with torch.no_grad():
     for xx, yy, zz in test_loader_pred2:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        zz = zz.to(device)
        for t in range(0, T_out, step):
            z = zz[:,:,:,t:t+step]
            im = model2(xx,yy)         
            loss += myloss(im.reshape(1, -1), z.reshape(1, -1))
            if t == 0:
               pred = im
            else:
               pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., 1:], yy[:,:,:,-1:]), dim=-1)
            yy = torch.cat((yy[..., 1:], im), dim=-1)
            
        
      #   pred0[index,:,:,:] = pred
        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(1, -1), zz.reshape(1, -1)).item()
        test_e.append(test_l2_step/ (T_out/step))
        
        print(index, test_l2_step/ (T_out/step), test_l2_full)
        index = index + 1
t2 = default_timer()
print(t2-t1)
test_e = torch.tensor((test_e))          
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')



