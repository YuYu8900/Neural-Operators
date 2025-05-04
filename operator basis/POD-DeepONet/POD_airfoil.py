import sys

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy import io

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa
import torch

def get_data(ntrain, ntest):
    # ntrain = 1000
    # ntest = 200

    INPUT_X = 'naca/NACA_Cylinder_X.npy'
    INPUT_Y = 'naca/NACA_Cylinder_Y.npy'
    OUTPUT_Sigma = 'naca/NACA_Cylinder_Q.npy'
        inputX = np.load(INPUT_X).astype(np.float32)
    inputY = np.load(INPUT_Y).astype(np.float32)
    input = np.stack([inputX, inputY], axis=3)

    r1 = 1
    r2 = 1
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    output = np.load(OUTPUT_Sigma)[:, 4].astype(np.float32)
    # output = torch.tensor(output, dtype=torch.float)

    print(input.shape, output.shape)
    x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2].reshape(-1, s1 * s2)
    y_train = np.vstack((np.ravel(y_train), np.ravel(y_train))).reshape(ntrain, s1*s2*2)
    x_test = input[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2].reshape(-1, s1 * s2)
    y_test = np.vstack((np.ravel(y_test), np.ravel(y_test))).reshape(ntest, s1*s2*2)
    x_train = x_train.reshape(ntrain, s1*s2*2)
    x_test = x_test.reshape(ntest, s1*s2*2)

    size_x = s1
    size_y = s2
    xx = np.linspace(0, 1, size_x).reshape(1,size_x)
    yy = np.linspace(0, 1, size_y).reshape(1,size_y)
    xx = np.repeat(xx.T,size_y,axis=1)
    xx = np.vstack((np.ravel(xx), np.ravel(xx)))
    yy = np.repeat(yy,size_x,axis=0)
    yy = np.vstack((np.ravel(yy), np.ravel(yy)))
    xy = np.vstack((np.ravel(xx), np.ravel(yy))).T
    
    return (x_train, xy), y_train,(x_test, xy), y_test


def pod(y):
    n = len(y)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (n - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    # w_cumsum = np.cumsum(w)
    # print(w_cumsum[:16] / w_cumsum[-1])
    # plt.figure()
    # plt.plot(y_mean)
    # plt.figure()
    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     plt.plot(v[:, i])
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
            x = tf.einsum("bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1))
            x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


def main():
    x_train, y_train, x_test, y_test = get_data(1000, 200)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    y_mean, v = pod(y_train)

    modes = 215
    m =  221*51*2
    activation = "relu"
    branch = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((221,51*2, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dense(modes),
        ]
    )
    branch.summary()
    net = PODDeepONet(v[:, :modes], [m, branch], None, activation, "Glorot normal")

    def output_transform(inputs, outputs):
        return outputs / modes + y_mean

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        tfa.optimizers.AdamW(1e-4, learning_rate=3e-4),
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=100000, batch_size=None)

    # dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname="burger_dis_loss.txt",output_dir='/home/fcx/yy/deeponet-fno-main/deeponet-fno-main/src/burgers/result')

    y_pred = model.predict(x_test)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_test, y_pred))



if __name__ == "__main__":
    main()
