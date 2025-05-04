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

    PATH_Sigma = 'elasticity/Meshes/Random_UnitCell_sigma_10.npy'
    PATH_XY = 'elasticity/Meshes/Random_UnitCell_XY_10.npy'
    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)
    train_s = input_s[:ntrain].reshape(ntrain,972)
    train_s = np.vstack((np.ravel(train_s), np.ravel(train_s))).reshape(ntrain, 972*2)
    test_s = input_s[-ntest:].reshape(ntest,972)
    test_s = np.vstack((np.ravel(test_s), np.ravel(test_s))).reshape(ntest, 972*2)
    train_xy = input_xy[:ntrain].reshape(ntrain,972*2)
    test_xy = input_xy[-ntest:].reshape(ntest,972*2)

    size_x = input_s.shape[1]
    size_y = input_s.shape[1]
    xx = np.linspace(0, 1, size_x).reshape(1,size_x)
    yy = np.linspace(0, 1, size_y).reshape(1,size_y)
    # xx = np.vstack((np.ravel(xx), np.ravel(xx)))
    # yy = np.vstack((np.ravel(yy), np.ravel(yy)))
    xy = np.vstack((np.ravel(xx), np.ravel(yy))).T
    xy = xy.reshape(972*2,1)
    
    return (train_xy, xy), train_s,(test_xy, xy), test_s


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
    # nx = 221*51
    x_train, y_train, x_test, y_test = get_data(1000, 200)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    pca = PCA(n_components=0.9999).fit(y_train)
    print("# Components:", pca.n_components_)

    net = PODDeepONet(
        pca.components_.T * 40,
        [972*2, 512, pca.n_components_],
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
