import numpy as onp
import scipy.io
from scipy.interpolate import griddata
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
# from jax.config import config
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
        branch_params = self.branch_init(rng_key = random.PRNGKey(2023))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(3202))
        params = (branch_params, trunk_params)

        # yy:weight
        self.data_weight = 1.0
        self.res_weight = 0.0
        self.ic_weight = 0.0
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
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, x, t):
        branch_params, trunk_params = params
        y = np.stack([x,t])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return   outputs
    
    # Define ds/dx
    def s_x_net(self, params, u, x, t):
         s_x = grad(self.operator_net, argnums=2)(params, u, x, t)
         return s_x

    # Define PDE residual        
    def residual_net(self, params, u, x, t):
        s = self.operator_net(params, u, x, t)
        s_x = grad(self.operator_net, argnums=2)(params, u, x, t)
        s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        s_xx= grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, t)

        res = s_t + s * s_x - 0.1 * s_xx
        return res

    # Define initial loss
    def loss_ics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])

        # Compute loss
        loss = np.mean((outputs.flatten() - s_pred)**2)
        return loss

    # Define boundary loss
    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_bc1_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        s_bc2_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,2], y[:,3])

        s_x_bc1_pred = vmap(self.s_x_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        s_x_bc2_pred = vmap(self.s_x_net, (None, 0, 0, 0))(params, u, y[:,2], y[:,3])

        # Compute loss
        loss_s_bc = np.mean((s_bc1_pred - s_bc2_pred)**2)
        loss_s_x_bc = np.mean((s_x_bc1_pred - s_x_bc2_pred)**2)

        return loss_s_bc + loss_s_x_bc

    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])

        # Compute loss
        loss = np.mean((outputs.flatten() - pred)**2)
        return loss   

    # Define data loss   
    def compute_data_loss(self, params, idx, usol, u0,m=101, P=101):
        u_train, y_train, s_train = generate_one_train_data(idx, usol,u0, m, P)

        u_train = u_train.reshape(m*P,-1)  
        y_train = y_train.reshape(m*P,-1)
        s_train = s_train.reshape(m*P,-1)

        s_pred = model.predict_s(params, u_train, y_train)[:,None]
        error = np.linalg.norm(s_train - s_pred) / np.linalg.norm(s_train)
        return error  
    
    def loss_data(self, params, idx, usol,u0, m=101, P=101):
        errors_data = vmap(self.compute_data_loss, in_axes=(None, 0, None, None, None,None))(params,idx, usol,u0, m, P)
        loss =  errors_data.mean()
        return loss 
        
    # Define total loss
    def loss(self, params,idx,usol,u0,ics_batch, bcs_batch, res_batch):
        # errors_data = vmap(self.loss_data, in_axes=(0, None, None, None))(params,idx, usol, m=128, P_train=101)
        loss_data = self.loss_data(params, idx, usol,u0)
        if self.res_weight == 0:
            loss = loss_data
        else:
            loss_ics = self.loss_ics(params, ics_batch)
            loss_bcs = self.loss_bcs(params, bcs_batch)
            loss_res = self.loss_res(params, res_batch)

            loss = self.data_weight*loss_data + self.ic_weight * loss_ics + self.bc_weight*loss_bcs + self.res_weight*loss_res #
        
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, idx, usol, u0, ics_batch, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, idx, usol, u0, ics_batch, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, usol,u0, ics_dataset, bcs_dataset, res_dataset, nIter = 10000):
        # train_data = iter(train_dataset)
        ics_data = iter(ics_dataset)
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        k0= 0
        N_train = 40
        # Main training loop
        for it in pbar:
            # Fetch data
            # train_batch= next(train_data)
            ics_batch= next(ics_data)
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)

            idx = np.arange(k0, k0 + N_train)
            self.opt_state = self.step(next(self.itercount), self.opt_state, idx, usol, u0, ics_batch, bcs_batch, res_batch)
            
            k0 = (k0+N_train)%1000
            if it % 1000 ==0:
                flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
                np.save('Burger/Burger_data_ex_params2.npy', flat_params)
                np.save('Burger/Burger_data_ex_losses2.npy', np.array([model.loss_log]))
            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, idx, usol, u0,ics_batch, bcs_batch, res_batch)
                loss_data_value = self.loss_data(params, idx, usol,u0)
                loss_ics_value = self.loss_ics(params, ics_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_data_log.append(loss_data_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value, 
                                  'loss_data':loss_data_value,
                                  'loss_ics' : loss_ics_value,
                                  'loss_bcs' : loss_bcs_value, 
                                  'loss_physics': loss_res_value})
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return r_pred
    
    # Geneate ics training data corresponding to one input sample
def generate_one_ics_training_data(key, u0, m=101, P=101):

    t_0 = np.zeros((P,1))
    x_0 = np.linspace(0, 1, P)[:, None]

    y = np.hstack([x_0, t_0])
    u = np.tile(u0, (P, 1))
    s = u0

    return u, y, s

# Geneate bcs training data corresponding to one input sample
def generate_one_bcs_training_data(key, u0, m=101, P=100):

    t_bc = random.uniform(key, (P,1))
    # t_bc = np.linspace(0, 1, P)
    x_bc1 = np.zeros((P, 1))
    x_bc2 = np.ones((P, 1))
  
    y1 = np.hstack([x_bc1, t_bc])  # shape = (P, 2)
    y2 = np.hstack([x_bc2, t_bc])  # shape = (P, 2)

    u = np.tile(u0, (P, 1))
    y =  np.hstack([y1, y2])  # shape = (P, 4)
    s = np.zeros((P, 1))
    # s = x.reshape(P,1)

    return u, y, s

# Geneate res training data corresponding to one input sample
def generate_one_res_training_data(key, u0, m=101, P=1000):

    subkeys = random.split(key, 2)
   
    t_res = random.uniform(subkeys[0], (P,1))
    x_res = random.uniform(subkeys[1], (P,1))

    u = np.tile(u0, (P, 1))
    y =  np.hstack([x_res, t_res])
    s = np.zeros((P, 1))

    return u, y, s

def generate_one_train_data(idx,usol,u0, m=128, P=101):
    u = usol[idx]
    u0 = u0[idx]

    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, m)
    X, T = np.meshgrid(x, t)

    s = u.T.flatten()
    u = np.tile(u0, (m*P, 1))
    y = np.hstack([X.flatten()[:,None], T.flatten()[:,None]])

    return u, y, s 

# Geneate test data corresponding to one input sample
def generate_one_test_data(idx,usol,u0, m=101, P=101):

    u = usol[idx]
    u0 = u0[idx]

    t = np.linspace(0, 1, P)
    x = np.linspace(0, 1, m)
    X, T = np.meshgrid(x, t)

    s = u.T.flatten()
    u = np.tile(u0, (m*P, 1))
    y = np.hstack([X.flatten()[:,None], T.flatten()[:,None]])

    return u, y, s 

# Geneate training data corresponding to N input sample
def compute_error(idx, usol,u0, m, P):
    u_test, y_test, s_test = generate_one_test_data(idx, usol,u0, m, P)

    u_test = u_test.reshape(m*P,-1)  
    y_test = y_test.reshape(m*P,-1)
    s_test = s_test.reshape(m*P,-1)

    s_pred = model.predict_s(params, u_test, y_test)[:,None]
    error = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test) 

    return error  
# Prepare the training data
N_train =1000      # number of input samples used for training
N_test = 100  # number of input samples used for test
m = 101            # number of sensors for input samples
P_ics_train = 101 # number of locations for evulating the initial condition
P_bcs_train = 100    # number of locations for evulating the boundary condition
P_res_train = 2500   # number of locations for evulating the PDE residual
P_test = 101

# Load data
data = np.load('ex_data/burgers/burgers_train_ls_1.0_101_101.npz')
usol = np.array( data['y_train'])[:N_train,:].reshape(1000,101,101)
# x = usol[0,0,:]
# x1 = usol[100,-1,:]
u0_train = np.array( data['X_train0'])[:N_train,:]


# P_train = 101       

# u0_train = usol[:N_train,0,:]   # input samples
# usol_train = usol[:N_train,:,:]
test_data = np.load('ex_data/burgers/burgers_test_ls_1.0_101_101.npz')
usol_test = np.array(test_data['y_test']).reshape(100,101,101)
u0_test = np.array(test_data['X_test0'])

test_data1 = np.load('ex_data/burgers/burgers_test_ls_0.6_101_101.npz')
usol_test1 = np.array(test_data1['y_test']).reshape(100,101,101)
u0_test1 = np.array(test_data1['X_test0'])


key = random.PRNGKey(0) # use different key for generating test data 
keys = random.split(key, N_train)


u_ics_train, y_ics_train, s_ics_train = vmap(generate_one_ics_training_data, in_axes=(0, 0, None, None))(keys, u0_train, m, P_ics_train)

u_ics_train = u_ics_train.reshape(N_train * P_ics_train,-1)  
y_ics_train = y_ics_train.reshape(N_train * P_ics_train,-1)
s_ics_train = s_ics_train.reshape(N_train * P_ics_train,-1)

# Generate training data for boundary condition
u_bcs_train, y_bcs_train, s_bcs_train = vmap(generate_one_bcs_training_data, in_axes=(0, 0, None, None))(keys, u0_train, m, P_bcs_train)

u_bcs_train = u_bcs_train.reshape(N_train * P_bcs_train,-1)  
y_bcs_train = y_bcs_train.reshape(N_train * P_bcs_train,-1)
s_bcs_train = s_bcs_train.reshape(N_train * P_bcs_train,-1)

# Generate training data for PDE residual
u_res_train, y_res_train, s_res_train = vmap(generate_one_res_training_data, in_axes=(0, 0, None, None))(keys, u0_train, m, P_res_train)

u_res_train = u_res_train.reshape(N_train * P_res_train,-1)  
y_res_train = y_res_train.reshape(N_train * P_res_train,-1)
s_res_train = s_res_train.reshape(N_train * P_res_train,-1)

# Initialize model
branch_layers = [m, 100, 100, 100, 100, 100, 100, 100]
trunk_layers =  [2, 100, 100, 100, 100, 100, 100, 100]
model = PI_DeepONet(branch_layers, trunk_layers)

# Create data set
batch_size = 50000
# train_dataset = DataGenerator(u_train, y_train, s_train, batch_size)
ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

# Train
# Note: may meet OOM issue if use Colab. Please train this model on the server.  
model.train(usol,u0_train,ics_dataset, bcs_dataset, res_dataset, nIter=50000)

# Save the trained model and losses
flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
np.save('Burger/Burger_data_ex_params2.npy', flat_params)
np.save('Burger/Burger_data_ex_losses2.npy', np.array([model.loss_log]))

# # Restore the trained model
params = model.get_params(model.opt_state)


# Compute relative l2 error over test data
k= 0
N_test = 100
idx = np.arange(k, k + N_test)

errors = vmap(compute_error, in_axes=(0, None, None, None, None))(idx, usol_test, u0_test, m, P_test)
mean_error = errors.mean()

print('Mean relative L2 error of s: {:.4e}'.format(mean_error))

k= 0
N_test = 100
idx = np.arange(k, k + N_test)

errors = vmap(compute_error, in_axes=(0, None, None, None, None))(idx, usol_test1, u0_test1, m, P_test)
mean_error = errors.mean()

print('Mean relative L2 error of s: {:.4e}'.format(mean_error))