import os
# print(torch.cuda.device_count())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #（保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  #（代表仅使用第0，1号GPU）
import numpy as onp
import h5py
from scipy.interpolate import griddata
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.config import config
from jax.flatten_util import ravel_pytree
from jax.nn import relu, elu
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange

# Define MLP
def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# Define modified MLP
def modified_MLP(layers, activation=relu):
  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = np.zeros(d_out)
      return W, b

  def init(rng_key):
      U1, b1 =  xavier_init(random.PRNGKey(12345), layers[0], layers[1])
      U2, b2 =  xavier_init(random.PRNGKey(54321), layers[0], layers[1])
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          W, b = xavier_init(k1, d_in, d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return (params, U1, b1, U2, b2) 

  def apply(params, inputs):
      params, U1, b1, U2, b2 = params
      U = activation(np.dot(inputs, U1) + b1)
      V = activation(np.dot(inputs, U2) + b2)
      for W, b in params[:-1]:
          outputs = activation(np.dot(inputs, W) + b)
          inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V) 
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:]
        y = self.y[idx,:]
        u = self.u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs
    
    # Define Physics-informed DeepONet model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):    
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = modified_MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = modified_MLP(trunk_layers, activation=np.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # yy:weight
        self.data_weight = 0.0
        self.res_weight = 1.0
        self.bc_weight = 0.0

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=2000, 
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_data_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, x, y, t):
        branch_params, trunk_params = params
        y = np.stack([x, y, t])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return   outputs
    
    # Define ds/dx
    def s_x_net(self, params, u, t, x):
         s_x = grad(self.operator_net, argnums=3)(params, u, t, x)
         return s_x

    # Define PDE residual        
    def residual_net(self, params, u, x, y, t):
        s = self.operator_net(params, u, x, y, t)
        s_t = grad(self.operator_net, argnums=4)(params, u, x, y, t)
        s_xx= grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, y, t)
        s_yy= grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, x, y, t)
        s_tt= grad(grad(self.operator_net, argnums=4), argnums=4)(params, u, x, y, t)

        res = s_t - (0.01 * (s_xx + s_yy) + s - s**3)
        return res

    # Define boundary loss
    def loss_bcs(self, params, idx, usol, u0, x, t):
        # Fetch data
        u, y, s = generate_one_train_data(idx, usol, u0, x, t)

        # Compute forward pass
        u = u.reshape(m*m*T,-1)  
        y = y.reshape(m*m*T,-1)
        s = s.reshape(m,m,T)

        # Compute forward pass
        pred =  model.predict_s(params, u, y)[:,None]
        pred = pred.reshape(m,m,T)

        s1 = s[0,:,:].flatten()
        s2 = s[-1,:,:].flatten()
        s3 = s[:,0,:].flatten()
        s4 = s[:,-1,:].flatten()
        p1 = pred[0,:,:].flatten()
        p2 = pred[-1,:,:].flatten()
        p3 = pred[:,0,:].flatten()
        p4 = pred[:,-1,:].flatten()
        s = np.vstack([s1, s2, s3, s4])
        s_pred = np.vstack([p1, p2, p3, p4])
        # Compute loss
        loss_bc = np.mean((s_pred - s)**2)

        return loss_bc

    # Define residual loss
    def loss_res(self, params, idx, usol, u0, x, t):
        # Fetch data
        u, y, s = generate_one_train_data(idx, usol, u0, x, t)

        u = u.reshape(m*m*T,-1)  
        y = y.reshape(m*m*T,-1)
        s = s.reshape(m*m*T,-1)

        # Compute forward pass
        pred = vmap(self.residual_net,(None, 0, 0, 0, 0))(params, u, y[:,0], y[:,1],y[:,2])

        # Compute loss
        loss = np.mean((pred)**2)
        return loss   

    # Define data loss   
    def compute_data_loss(self, params,  idx, usol, u0, x, t):
        u_train, y_train, s_train = generate_one_train_data( idx, usol, u0, x, t)

        u_train = u_train.reshape(m*m*T,-1)  
        y_train = y_train.reshape(m*m*T,-1)
        s_train = s_train.reshape(m*m*T,-1)

        s_pred = model.predict_s(params, u_train, y_train)[:,None]
        error = np.linalg.norm(s_train - s_pred) / np.linalg.norm(s_train)
        return error  
    
    def loss_data(self, params, idx, usol, u0, x, t):
        errors_data = vmap(self.compute_data_loss, in_axes=(None, 0, None, None, None,None))(params, idx, usol, u0, x, t)
        loss =  errors_data.mean()
        return loss 
        
    # Define total loss
    def loss(self, params, idx, usol, u0, x, t):
        # errors_data = vmap(self.loss_data, in_axes=(0, None, None, None))(params,idx, usol, m=128, P_train=101)
        loss_data = self.loss_data(params,idx, usol, u0, x, t)
        if self.res_weight == 0:
            loss = loss_data
        else:
            loss_bcs = vmap(self.loss_bcs, in_axes=(None, 0, None, None, None))(params, idx, usol, u0, x, t).mean()
            loss_res = vmap(self.loss_res, in_axes=(None, 0, None, None, None))(params, idx, usol, u0, x, t).mean()
            loss = self.data_weight*loss_data + self.bc_weight*loss_bcs + self.res_weight*loss_res #
        
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state,  idx, usol, u0, x, t):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params,  idx, usol, u0, x, t)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, usol, u0, x, t, nIter = 10000):
        pbar = trange(nIter)
        k0= 0
        N_train = 20
        # Main training loop
        for it in pbar:
            idx = np.arange(k0, k0 + N_train)
            self.opt_state = self.step(next(self.itercount), self.opt_state, idx, usol, u0, x, t)
            
            k0 = (k0+N_train)%600
            if it % 1000 ==0:
                flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
                np.save('AC_data_params.npy', flat_params)
                np.save('AC_data_losses.npy', np.array([model.loss_log]))
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, idx, usol, u0, x, t)
                loss_data_value = self.loss_data(params, idx, usol, u0, x, t)
                loss_bcs_value = vmap(self.loss_bcs, in_axes=(None, 0, None, None, None, None))(params, idx, usol, u0, x, t).mean()
                loss_res_value = vmap(self.loss_res, in_axes=(None, 0, None, None, None, None))(params, idx, usol, u0, x, t).mean()

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_data_log.append(loss_data_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value, 
                                  'loss_data':loss_data_value,
                                  'loss_bcs' : loss_bcs_value, 
                                  'loss_physics': loss_res_value})
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1], Y_star[:,2])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1], Y_star[:,2])
        return r_pred

# Geneate bcs training data corresponding to one input sample
def generate_one_bcs_training_data(idx,usol,u0,x,t):
    u = usol[idx]
    u0 = u0[idx]
    t_bc = np.tile(t[0:T_in],(m,1))
    x_bc1 = np.tile(np.zeros((m, 1)),(1,T))
    y_bc1 =  np.tile(x,(1,T))

    x_bc2 = np.tile(np.ones((m, 1)),(1,T))
    y_bc2 = np.tile(x,(1,T))

    x_bc3 = np.tile(x,(1,T))
    y_bc3 = np.tile(np.zeros((m, 1)),(1,T))

    x_bc4 = np.tile(x,(1,T))
    y_bc4 = np.tile(np.ones((m, 1)),(1,T))

    y1 = np.hstack([t_bc.flatten()[:,None], x_bc1.flatten()[:,None], y_bc1.flatten()[:,None]])  
    y2 = np.hstack([t_bc.flatten()[:,None], x_bc2.flatten()[:,None], y_bc2.flatten()[:,None]])
    y3 = np.hstack([t_bc.flatten()[:,None], x_bc3.flatten()[:,None], y_bc3.flatten()[:,None]]) 
    y4 = np.hstack([t_bc.flatten()[:,None], x_bc4.flatten()[:,None], y_bc4.flatten()[:,None]])

    u = np.tile(u0.flatten()[:,None], (m*T, 1))
    y =  np.hstack([y1, y2, y3, y4])  # shape = (P, 4)
    s1 = u[0,:,:].flatten()
    s2 = u[-1,:,:].flatten()
    s3 = u[:,0,:].flatten()
    s4 = u[:,-1,:].flatten()
    s = np.vstack([s1, s2, s3, s4])

    return u, y, s

# # Geneate res training data corresponding to one input sample
# def generate_one_res_training_data(key, u0, m=101, P=1000):

#     subkeys = random.split(key, 2)
   
#     t_res = random.uniform(subkeys[0], (P,1))
#     x_res = random.uniform(subkeys[1], (P,1))

#     u = np.tile(u0, (P, 1))
#     y =  np.hstack([t_res, x_res])
#     s = np.zeros((P, 1))

#     return u, y, s

def generate_one_train_data(idx,usol,u0,x,t):
    u = usol[idx]
    u0 = u0[idx]
    t = t[0:T_in]
    X, Y, T = np.meshgrid(x, x, t)

    s = u.flatten()
    u0 = u0.flatten()
    N = 10890
    u = np.tile(u0, (N,1))
    y = np.hstack([X.flatten()[:,None],Y.flatten()[:,None], T.flatten()[:,None]])

    return u, y, s 

# Geneate test data corresponding to one input sample
def generate_one_test_data(idx,usol,u0,x,t):

    u = usol[idx]
    u0 = u0[idx]
    t = t[0:T_in]
    X, Y, T = np.meshgrid(x, x, t)

    s = u.flatten()
    u0 = u0.flatten()
    N = 10890
    u = np.tile(u0, (N,1))
    y = np.hstack([X.flatten()[:,None],Y.flatten()[:,None], T.flatten()[:,None]])

    return u, y, s 

# Geneate training data corresponding to N input sample
def compute_error(idx,usol,u0,x,t):
    u_test, y_test, s_test = generate_one_test_data(idx,usol,u0,x,t)

    u_test = u_test.reshape(m*m*T,-1)  
    y_test = y_test.reshape(m*m*T,-1)
    s_test = s_test.reshape(m*m*T,-1)

    s_pred = model.predict_s(params, u_test, y_test)[:,None]
    error = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test) 

    return error  


# Prepare the training data

# Load data
path = 'Allen_Cahn_pde_65_65_1000.mat'#'Burger.mat'  # Please use the matlab script to generate data

data = h5py.File(path)
usol = np.array( data['sol'])
usol = usol.transpose(0,2,3,1)
t = np.array( data['t']).reshape(51)
X = np.array( data['x']).reshape(65)


N = usol.shape[0]  # number of total input samples
N_train =600      # number of input samples used for training
N_test = N - N_train  # number of input samples used for test
m = 33           # resolution
T_in = 10
T = 10
sub = 2

x = X[::sub]
u0_train = usol[:N_train,::sub,::sub,0]   # input samples
usol_train = usol[:N_train,::sub,::sub,:T_in]


# Initialize model
branch_layers = [m*m, 100, 100, 100, 100, 100, 100, 100]
trunk_layers =  [3, 100, 100, 100, 100, 100, 100, 100]
model = PI_DeepONet(branch_layers, trunk_layers)

# Train
# Note: may meet OOM issue if use Colab. Please train this model on the server.  
model.train(usol_train, u0_train, x, t, nIter=200000)

# Save the trained model and losses
flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
np.save('AC_data_params.npy', flat_params)
np.save('AC_data_losses.npy', np.array([model.loss_log]))

# Restore the trained model
params = model.get_params(model.opt_state)


# Compute relative l2 error over test data
k= 600
N_test = 20
idx = np.arange(k, k + N_test)
u0_test = usol[:,:,:,:T_in]

errors = vmap(compute_error, in_axes=(0, None, None, None))(idx, usol, u0_test, x, t)
mean_error = errors.mean()

print('Mean relative L2 error of s: {:.4e}'.format(mean_error))