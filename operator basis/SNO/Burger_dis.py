import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import jax.numpy as jnp
from jax import random
from architectures import SNO_2D as vanilla
from functions import Chebyshev,Fourier, utils
from scipy import io
import numpy as np
import optax


ntrain = 480
ntest = 20
s = 512

sub_x = 1
sub_y = 1
T_in = 20
T = 30

data = io.loadmat("burgers_data_512_51.mat")

u = data["sol"].astype(np.float32) # N x Nx x Nt
# val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0)), periodic=True), axes=(2, 0, 1))
# u = val_to_coeff(jnp.stack(u))

x_train = u[:ntrain, ::sub_x, :T_in].reshape(ntrain,s,T_in)
y_train = u[:ntrain, ::sub_x, T_in:T_in+T].reshape(ntrain,s,T)

x_test = u[-ntest:, ::sub_x, :T_in].reshape(ntest,s,T_in) 
y_test = u[-ntest:, ::sub_x, T_in:T_in+T].reshape(ntest,s,T)


val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0)), periodic=False), axes=(2, 0, 1))
x_train = val_to_coeff(jnp.stack(x_train)).reshape(ntrain,s,T_in,1)
x_test = val_to_coeff(jnp.stack(x_test)).reshape(ntest,s,T_in,1)
y_train = val_to_coeff(jnp.stack(y_train)).reshape(ntrain,s,T,1)
y_test = val_to_coeff(jnp.stack(y_test)).reshape(ntest,s,T,1)

params_encoder = vanilla.init_c_network_params([1, 10, 10], random.PRNGKey(11))
params_i = vanilla.init_i_network_params([512, 512, 512, 512], [T_in, T, T, T], [10, 10, 10, 10], random.PRNGKey(11))
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
    test_loss.append(l_test)
    train_loss.append(l_train)
    print("Epoch:%d,train loss:%f , test loss:%f"%(i,l_train,l_test))
    vanilla.save_params(params, "C_burgers_dis_params", "burgers_dis_params")  

coeff_to_val = lambda x: jnp.transpose(Chebyshev.coefficients_to_values(jnp.transpose(x, axes=(1, 2, 0))), axes=(2, 0, 1))
# params = vanilla.load_params("test/burgers_dis_params", "burgers_dis_params")
predictions = coeff_to_val(jnp.stack(vanilla.batched_NN(params, x_test).reshape(ntest,s,T))).reshape(ntest,-1)
targets = coeff_to_val(jnp.stack(y_test.reshape(ntest,s,T))).reshape(ntest,-1)
relative_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
mean_relative_test_error = jnp.mean(relative_errors)
print("Mean relative test error", mean_relative_test_error)
