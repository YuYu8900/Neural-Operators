import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import jax.numpy as jnp
from jax import random
from architectures import SNO_2D as vanilla
from functions import Chebyshev,Fourier, utils
from scipy import io
import numpy as np
import optax


TRAIN_PATH = 'ns_data_V100_N1000_T50_1.mat'
TEST_PATH = 'ns_data_V100_N1000_T50_2.mat'
ntrain = 1000
ntest = 20

sub = 1
S = 64
T_in = 10
T = 10
################################################################
# load data and data normalization
################################################################
data = io.loadmat(TRAIN_PATH)
train_a = data[:ntrain,::sub,::sub,:T_in]
train_u = data[:ntrain,::sub,::sub,T_in:T+T_in]
data1 = io.loadmat(TEST_PATH)
test_a = data[-ntest:,::sub,::sub,:T_in]
test_u = data[-ntest:,::sub,::sub,T_in:T+T_in]


val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 3, 0)), periodic=False), axes=(3, 0, 1, 2))
x_train = val_to_coeff(jnp.stack(train_a)).reshape(ntrain,S,S,T_in,1)
x_test = val_to_coeff(jnp.stack(test_a)).reshape(ntest,S,S,T_in,1)
y_train = val_to_coeff(jnp.stack(train_u)).reshape(ntrain,S,S,T,1)
y_test = val_to_coeff(jnp.stack(test_u)).reshape(ntest,S,S,T,1)

params_encoder = vanilla.init_c_network_params([1, 10, 10], random.PRNGKey(11))
params_i = vanilla.init_i_network_params([S, S, S, S], [S, S, S, S],[T_in, T_in, T_in, T_in],[10, 10, 10, 10], random.PRNGKey(11))
params_decoder = vanilla.init_c_network_params([10, 1], random.PRNGKey(11)) 
params = [params_encoder, params_i, params_decoder]

print(vanilla.count_params(params))

sc = optax.exponential_decay(0.001, 5000, 0.5)
optimizer = optax.adam(sc)
opt_state = optimizer.init(params)

utils.update_params(params, x_train, y_train, optimizer, opt_state, vanilla.loss)

N_epochs = 30000

train_loss = []
test_loss = []

test_loss.append(vanilla.loss(params, x_test, y_test))
train_loss.append(vanilla.loss(params, x_train, y_train))

print(train_loss[-1])
print(test_loss[-1])
for i in range(N_epochs):
  # run in a single batch
  params, opt_state = utils.update_params(params, x_train, y_train, optimizer, opt_state, vanilla.loss)
  if (i+1)%20 == 0:
    l_test = vanilla.loss(params, x_test, y_test)
    l_train = vanilla.loss(params, x_train, y_train)
    test_loss.append(l_test)
    train_loss.append(l_train)
    print("Epoch:%d,train loss:%f , test loss:%f"%(i,l_train,l_test))
    vanilla.save_params(params, "F_NS_params", "NS_params")  

params = vanilla.load_params("test/F_NS_params", "naca_params")
coeff_to_val = lambda x: jnp.transpose(Fourier.coefficients_to_values(jnp.transpose(x, axes=(1, 2, 3, 0)),(S,S,T,ntest)), axes=(3, 0, 1, 2))
predictions = coeff_to_val(jnp.stack(vanilla.batched_NN(params, x_test).reshape(ntest,S,S,T)))
targets = coeff_to_val(y_test.reshape(ntest,S,S,T))
relative_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
mean_relative_test_error = jnp.mean(relative_errors)
print("Mean relative test error", mean_relative_test_error)