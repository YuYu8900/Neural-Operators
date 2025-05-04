import jax.numpy as jnp
from jax import random
from architectures import fSNO_2D as vanilla
from functions import Chebyshev,Fourier, utils
from scipy import io
import numpy as np
import optax
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

ntrain = 1000
ntest = 200
s1 = 36
s2 = 27
################################################################
# load data and data normalization
################################################################
PATH_Sigma = 'elasticity/Meshes/Random_UnitCell_sigma_10.npy'
PATH_XY = 'elasticity/Meshes/Random_UnitCell_XY_10.npy'
# PATH_rr = 'elasticity/Meshes/Random_UnitCell_rr_10.npy'

# input_rr = np.load(PATH_rr)
# input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1,0)
input_s = np.load(PATH_Sigma).astype(np.float32)
input_s = np.transpose(input_s, (1,0))
input_xy = np.load(PATH_XY).astype(np.float32)
input_xy = input_xy.transpose(2,0,1)

train_s = input_s[:ntrain].reshape(ntrain,s1*s2)
test_s = input_s[-ntest:].reshape(ntest,s1*s2)
train_xy = input_xy[:ntrain].reshape(ntrain,s1*s2,2)
test_xy = input_xy[-ntest:].reshape(ntest,s1*s2,2)

# val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 3, 0)), periodic=False), axes=(3, 0, 1, 2))
val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0)), periodic=False), axes=(2, 0, 1))
x_train = val_to_coeff(jnp.stack(train_xy)).reshape(ntrain,s1*s2,2)
x_test = val_to_coeff(jnp.stack(test_xy)).reshape(ntest,s1*s2,2)
val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 0)), periodic=False), axes=(0, 1))
y_train = val_to_coeff(jnp.stack(train_s)).reshape(ntrain,s1*s2,1)
y_test = val_to_coeff(jnp.stack(test_s)).reshape(ntest,s1*s2,1)

params_encoder = vanilla.init_c_network_params([2, 10, 10], random.PRNGKey(11))
params_i = vanilla.init_i_network_params([s1*s2, s1*s2, s1*s2, s1*s2], [10, 10, 10, 10], random.PRNGKey(11))
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
    vanilla.save_params(params, "C_elsa_params", "elsa_params")  

params = vanilla.load_params("test/C_elsa_params", "elsa_params")
coeff_to_val = lambda x: jnp.transpose(Chebyshev.coefficients_to_values(jnp.transpose(x, axes=(1, 2, 0)),(s1,s2,ntest)), axes=(2, 0, 1))
predictions = coeff_to_val(jnp.stack(vanilla.batched_NN(params, x_test).reshape(ntest,s1,s2)))
targets = coeff_to_val(y_test.reshape(ntest,s1,s2))
relative_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
mean_relative_test_error = jnp.mean(relative_errors)
print("Mean relative test error", mean_relative_test_error)