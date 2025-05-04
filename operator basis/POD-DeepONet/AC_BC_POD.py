import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically
# sess = tf.Session(config = config)

def get_data(filename, ntrain,ntest):
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29
    r = 3
    h = int(((129 - 1)/r) + 1)
    s = h

    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    xtrain_branch = data["coeff"][:ntrain, ::r, ::r][:,:s,:s].astype(np.float32) 
    ytrain = data["sol"][:ntrain, ::r, ::r][:,:s,:s].astype(np.float32)
    xtest_branch = data['coeff'][-ntest:,::r,::r][:,:s,:s].astype(np.float32) 
    ytest = data["sol"][-ntest:,::r,::r][:,:s,:s].astype(np.float32) 

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    xtrain_branch = xtrain_branch.reshape(ntrain, s * s)
    xtest_branch = xtest_branch.reshape(ntest, s * s)
    xtrain = (xtrain_branch, grid)
    xtest = (xtest_branch, grid)
    ytrain = ytrain.reshape(ntrain, s * s)
    ytest = ytest.reshape(ntest, s * s)
    return xtrain, ytrain, xtest, ytest


def pod(y):
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (len(y) - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    # w_cumsum = np.cumsum(w)
    # print(w_cumsum[:115] / w_cumsum[-1])
    # plt.figure()
    # plt.imshow(y_mean.reshape(s, s))
    # plt.figure()
    # plt.imshow(v[:, 0].reshape(s, s))
    # plt.show()
    return y_mean, v


class PODDeepONet(dde.maps.NN):
    def __init__(
        self,
        pod_basis,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = dde.maps.activations.get(
                activation
            )

        self.pod_basis = tf.convert_to_tensor(pod_basis, dtype=tf.float32)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = dde.maps.FNN(
                layer_sizes_branch, activation_branch, kernel_initializer
            )
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = dde.maps.FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
        self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        x_func = self.branch(x_func)
        if self.trunk is None:
            # POD only
            x = tf.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            # y1=tf.concat((self.pod_basis, tf.transpose(x_loc)), 1)
            x = tf.concat((self.pod_basis, x_loc), 1)
            x = tf.einsum("bi,ni->bn", x_func, x)
            # x = tf.einsum("bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1))
            x = x + self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


def main():
    nx = 43
    x_train, y_train,x_test, y_test = get_data("Allen_cahn_2d_128_128_T10.mat", 1400,100)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
    y_mean, v = pod(y_train)

    modes = 20
    m = 43 ** 2
    activation = "relu"
    branch = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((43, 43, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dense(modes),
        ]
    )
    trunk = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Reshape((43, 43, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dense(modes),
        ]
    )
    branch.summary()
    net = PODDeepONet(v[:, :modes//2], [m, branch], [2, modes//2] ,  activation, "Glorot normal")
    # net = PODDeepONet(v[:, :modes], [m, branch],  None,  activation, "Glorot normal")

    def output_transform(inputs, outputs):
        return outputs / modes + y_mean

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        tfa.optimizers.AdamW(1e-4, learning_rate=3e-4),
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=100000, batch_size=10,display_every=100)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname="burger_dis_loss.txt",output_dir='/home/fcx/yy/deeponet-fno-main/deeponet-fno-main/src/burgers/result')

    y_pred = model.predict(x_test)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_test, y_pred))

if __name__ == "__main__":
    main()
