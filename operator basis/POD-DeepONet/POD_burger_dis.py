import sys

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy import io

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa

def get_data(ntrain, ntest):
    # sub_x = 2 ** 6
    # sub_y = 2 ** 6
    s = 512
    sub_x = 1
    sub_y = 1
    T_in = 20
    T = 30
    nx = 512
    nt =51
    data = io.loadmat('data/burgers_data_512_51.mat')
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["sol"].astype(np.float32) # N x Nx x Nt

    u0_train = u[:ntrain,:, 0]  # N x nx
    x = np.repeat(x.T,nt,axis=1)
    t = np.repeat(t,nx,axis=0)
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    u_train = u[:ntrain,:, :].reshape(-1, nt * nx)

    u0_test = u[-ntest:,:, 0]  # N x nx
    u_test = u[-ntest:,:, :].reshape(-1, nt * nx)
    
    return (u0_train, xt), u_train,(u0_test, xt), u_test
 

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
    nx = 512
    x_train, y_train, x_test, y_test = get_data(480, 20)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    pca = PCA(n_components=0.9999).fit(y_train)
    print("# Components:", pca.n_components_)
    # print(np.cumsum(pca.explained_variance_ratio_))
    # plt.figure()
    # plt.semilogy(pca.explained_variance_ratio_, 'o')
    # plt.figure()
    # plt.imshow(pca.mean_.reshape(nt, nx))
    # plt.colorbar()
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.imshow(pca.components_[i].reshape(nt, nx) * 40)
    #     plt.colorbar()
    # plt.show()
    net = PODDeepONet(
        pca.components_.T * 40,
        [nx, 512, pca.n_components_],
        None,
        "relu",
        "Glorot normal",
    )

    def output_transform(inputs, outputs):
        return outputs / pca.n_components_ + pca.mean_

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=100000, batch_size=None)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname="burger_dis_loss.txt",output_dir='/home/fcx/yy/deeponet-fno-main/deeponet-fno-main/src/burgers/result')

    y_pred = model.predict(x_test)
    print("L2 relative error:", dde.metrics.l2_relative_error(y_test, y_pred))
    # np.savetxt("burger_POD.dat", np.hstack((x_test, y_test, y_pred)))


if __name__ == "__main__":
    main()
