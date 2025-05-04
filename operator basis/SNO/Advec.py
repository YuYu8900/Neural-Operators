import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import jax.numpy as jnp
from jax import random
import optax
# from datasets import Indefinite_Integrals
from architectures import SNO_2D as vanilla
from functions import Chebyshev, utils
from scipy import io
import numpy as np


ntrain = 1000
ntest = 100
s = 40

T = 40

val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0)), periodic=False), axes=(2, 0, 1))
data = np.load("time_advection/train_IC2.npz")
u_train = data["u"] # N x Nt x Nx
u_train = val_to_coeff(jnp.stack(u_train))
x_train = u_train[:ntrain, 0, :].reshape(ntrain,s,1,1) # N x nx
y_train = u_train[:ntrain, :, :]
y_train = y_train.transpose(0, 2, 1).reshape(ntrain,s,s,1)

data1 = np.load("time_advection/test_IC2.npz")
u_test = data1["u"].astype(np.float32) # N x Nt x Nx
u_test = val_to_coeff(jnp.stack(u_test))
x_test = u_test[:ntest, 0, :].reshape(ntest,s,1,1) # N x nx
y_test = u_test[:ntest, :, :]
y_test = y_test.transpose(0, 2, 1).reshape(ntest,s,s,1)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

params_encoder = vanilla.init_c_network_params([1, 10, 10], random.PRNGKey(11))
params_i = vanilla.init_i_network_params([s, s, s, s], [1, T, T, T], [10, 10, 10, 10], random.PRNGKey(11))
params_decoder = vanilla.init_c_network_params([10, 1], random.PRNGKey(11)) 
params = [params_encoder, params_i, params_decoder]

print(vanilla.count_params(params))

sc = optax.exponential_decay(0.001, 5000, 0.5)
optimizer = optax.adam(sc)
opt_state = optimizer.init(params)

utils.update_params(params, x_train, y_train, optimizer, opt_state, vanilla.loss)

N_epochs = 50000

train_loss = []
test_loss = []

test_loss.append(vanilla.loss(params, x_test, y_test))
train_loss.append(vanilla.loss(params, x_train, y_train))

print(train_loss[-1])
print(test_loss[-1])
for i in range(N_epochs):
  # run in a single batch
  params, opt_state = utils.update_params(params, x_train, y_train, optimizer, opt_state, vanilla.loss)
  if (i+1)%50 == 0:
    l_test = vanilla.loss(params, x_test, y_test)
    l_train = vanilla.loss(params, x_train, y_train)
    p = vanilla.batched_NN(params, x_test)
    test_loss.append(l_test)
    train_loss.append(l_train)
    print("Epoch:%d,train loss:%f , test loss:%f"%(i,l_train,l_test))
    vanilla.save_params(params, "advec_params", "advec_params")  

# params = vanilla.load_params("test/advec_params", "advec_params")
coeff_to_val = lambda x: jnp.transpose(Chebyshev.coefficients_to_values(jnp.transpose(x, axes=(1, 2, 0))), axes=(2, 0, 1))
predictions = coeff_to_val(jnp.stack(vanilla.batched_NN(params, x_test).reshape(ntest,s,s))).reshape(ntest,-1)
targets = coeff_to_val(jnp.stack(y_test.reshape(ntest,s,s))).reshape(ntest,-1)
relative_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
mean_relative_test_error = jnp.mean(relative_errors)
print("Mean relative test error", mean_relative_test_error)