import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import jax.numpy as jnp
from jax import random
import optax
from architectures import SNO_1D as vanilla
from functions import Chebyshev,Fourier, utils
from scipy import io
import numpy as np

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
ntrain = 1000
ntest = 100

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

data = io.loadmat("burgers_data_R10.mat")
x_data = data["a"][:, ::sub].astype(np.float32)
y_data = data["u"][:, ::sub].astype(np.float32)
x_train = x_data[:ntrain, :]
y_train = y_data[:ntrain, :]
x_test = x_data[-ntest:, :]
y_test = y_data[-ntest:, :]
val_to_coeff = lambda x: utils.values_to_coefficients(x.T, periodic=False).T
x_train = val_to_coeff(jnp.stack(x_train)).reshape(ntrain,s,1)
y_train = val_to_coeff(jnp.stack(y_train)).reshape(ntrain,s,1)
x_test = val_to_coeff(jnp.stack(x_test)).reshape(ntest,s,1)
y_test = val_to_coeff(jnp.stack(y_test)).reshape(ntest,s,1)


# initialize network with 3 layers
params_encoder = vanilla.init_c_network_params([1, 10, 10], random.PRNGKey(11))
#  init_i_network_params(sizes, c_sizes, key)
params_i = vanilla.init_i_network_params([s, s, s, s], [10, 10, 10, 10], random.PRNGKey(11))
params_decoder = vanilla.init_c_network_params([10, 1], random.PRNGKey(11))
params = [params_encoder, params_i, params_decoder]
print(vanilla.count_params(params))

# initialize optimizer
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
  if (i+1)%100 == 0:
    l_test = vanilla.loss(params, x_test, y_test)
    l_train = vanilla.loss(params, x_train, y_train)
    test_loss.append(vanilla.loss(params,x_test, y_test))
    train_loss.append(vanilla.loss(params, x_train, y_train))
    print("Epoch:%d,train loss:%f , test loss:%f"%(i,l_train,l_test))
    vanilla.save_params(params, "C_burgers_params", "burgers_params") 

coeff_to_val = lambda x: Chebyshev.coefficients_to_values(x.T).T
# params = vanilla.load_params("Cburgers_params", "Cburgers_params")
l_test = vanilla.loss(params, x_test, y_test)
predictions = coeff_to_val(jnp.stack(vanilla.batched_NN(params, x_test).reshape(ntest,s)))
targets = coeff_to_val(jnp.stack(y_test.reshape(ntest,s)))
relative_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
mean_relative_test_error = jnp.mean(relative_errors)
print("Mean relative test error", mean_relative_test_error)

