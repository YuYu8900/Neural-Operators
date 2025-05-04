import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import jax.numpy as jnp
from jax import random
from architectures import SNO_2D as vanilla
from functions import Chebyshev, utils
from scipy import io
import numpy as np
import optax


TRAIN_PATH = 'Darcy_421/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'Darcy_421/piececonst_r421_N1024_smooth2.mat'
ntrain = 1000
ntest = 200

r1 = 5
r2 = 5
s1 = int(((421 - 1) / r1) + 1)
s2 = int(((421 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################
data = io.loadmat(TRAIN_PATH)
x_train = data['coeff'][:ntrain,::r1,::r2][:,:s1,:s2].astype(np.float32)
y_train = data['sol'][:ntrain,::r1,::r2][:,:s1,:s2].astype(np.float32)
    
data2 = io.loadmat(TEST_PATH)
x_test = data['coeff'][:ntest,::r1,::r2][:,:s1,:s2].astype(np.float32)
y_test = data['sol'][:ntest,::r1,::r2][:,:s1,:s2].astype(np.float32)

val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0)), periodic=False), axes=(2, 0, 1))
x_train = val_to_coeff(jnp.stack(x_train)).reshape(ntrain,s1,s2,1)
x_test = val_to_coeff(jnp.stack(x_test)).reshape(ntest,s1,s2,1)
y_train = val_to_coeff(jnp.stack(y_train)).reshape(ntrain,s1,s2,1)
y_test = val_to_coeff(jnp.stack(y_test)).reshape(ntest,s1,s2,1)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
params_encoder = vanilla.init_c_network_params([1, 10, 10], random.PRNGKey(11))
params_i = vanilla.init_i_network_params([s1, s1, s1, s1], [s2, s2, s2, s2], [10, 10, 10, 10], random.PRNGKey(11))
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
  if (i+1)%20 == 0:
    l_test = vanilla.loss(params, x_test, y_test)
    l_train = vanilla.loss(params, x_train, y_train)
    test_loss.append(l_test)
    train_loss.append(l_train)
    print("Epoch:%d,train loss:%f , test loss:%f"%(i,l_train,l_test))
    vanilla.save_params(params, "darcy_params", "darcy_params")  

# params = vanilla.load_params("test/advec_params", "advec_params")
coeff_to_val = lambda x: jnp.transpose(Chebyshev.coefficients_to_values(jnp.transpose(x, axes=(1, 2, 0))), axes=(2, 0, 1))
predictions = coeff_to_val(jnp.stack(vanilla.batched_NN(params, x_test).reshape(ntest,s1,s2))).reshape(ntest,-1)
targets = coeff_to_val(jnp.stack(y_test.reshape(ntest,s1,s2))).reshape(ntest,-1)
relative_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
mean_relative_test_error = jnp.mean(relative_errors)
print("Mean relative test error", mean_relative_test_error)