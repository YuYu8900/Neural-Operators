import jax.numpy as jnp
from jax import random
from architectures import SNO_2D as vanilla
from functions import Chebyshev,Fourier, utils
from scipy import io
import numpy as np
import optax
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

PATH = ""
INPUT_X = PATH+'/data/naca/NACA_Cylinder_X.npy'
INPUT_Y = PATH+'/data/naca/NACA_Cylinder_Y.npy'
OUTPUT_Sigma = PATH+'/data/naca/NACA_Cylinder_Q.npy'

ntrain = 1000
ntest = 200


r1 = 1
r2 = 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################
inputX = np.load(INPUT_X).astype(np.float32)
inputY = np.load(INPUT_Y).astype(np.float32)

val_to_coeff = lambda x: jnp.transpose(utils.values_to_coefficients(jnp.transpose(x, axes=(1, 2, 0)), periodic=False), axes=(2, 0, 1))
inputx = val_to_coeff(jnp.stack(inputX[:, ::r1, ::r2][:, :s1, :s2]))
inputy = val_to_coeff(jnp.stack(inputY[:, ::r1, ::r2][:, :s1, :s2]))

input = np.stack([inputx, inputy], axis=3)

output = np.load(OUTPUT_Sigma)[:, 4].astype(np.float32)
output = output[:, ::r1, ::r2][:, :s1, :s2]
output = val_to_coeff(jnp.stack(output))

x_train = input[:ntrain,:,:,:]
x_test = input[ntrain:ntrain+ntest,:,:,:]
y_train = output[:ntrain,:,:].reshape(ntrain,s1,s2,1)
y_test = output[ntrain:ntrain+ntest,:,:].reshape(ntest,s1,s2,1)
print(input.shape, output.shape)

params_encoder = vanilla.init_c_network_params([2, 10, 10], random.PRNGKey(11))
params_i = vanilla.init_i_network_params([s1, s1, s1, s1], [s2, s2, s2, s2], [10, 10, 10, 10], random.PRNGKey(11))
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
    vanilla.save_params(params, "C_airfoil_params", "naca_params")  

# params = vanilla.load_params("test/F_naca_params", "naca_params")
coeff_to_val = lambda x: jnp.transpose(Chebyshev.coefficients_to_values(jnp.transpose(x, axes=(1, 2, 0)),(s1,s2,ntest)), axes=(2, 0, 1))
predictions = coeff_to_val(jnp.stack(vanilla.batched_NN(params, x_test).reshape(ntest,s1,s2)))
targets = coeff_to_val(y_test.reshape(ntest,s1,s2))
relative_errors = jnp.linalg.norm(predictions - targets, axis=1) / jnp.linalg.norm(targets, axis=1)
mean_relative_test_error = jnp.mean(relative_errors)
print("Mean relative test error", mean_relative_test_error)