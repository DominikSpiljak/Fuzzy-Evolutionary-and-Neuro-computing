import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
import pickle


class ANFIS():

    def __init__(self, no_rules=1):
        self.p = np.random.randn(no_rules, 1)
        self.q = np.random.randn(no_rules, 1)
        self.r = np.zeros((no_rules, 1))

        self.a = np.random.randn(no_rules, 2)
        self.b = np.random.randn(no_rules, 2)

    def save_model(self, filepath):
        np.savez(filepath, p=self.p, q=self.q, r=self.r, a=self.a, b=self.b)

    @staticmethod
    def load_model(filepath):
        data = np.load(filepath)
        model = ANFIS()
        p = data['p']
        q = data['q']
        r = data['r']
        a = data['a']
        b = data['b']
        model.p = p
        model.q = q
        model.r = r
        model.a = a
        model.b = b

        return model

    def fuzzification_sigmoid(self, x_y):
        return 1 / (1 + np.exp(self.b * (x_y - self.a)))

    def mean_squared_err(self, z_true, preds):
        return np.mean(np.square(z_true - preds))

    def forward(self, X_Y, underflow_ctrl=1e-12):
        outs = []
        alphas = []

        for x_y in X_Y:

            # 1. layer: Fuzzification
            fuzzified = self.fuzzification_sigmoid(x_y)

            # 2. layer: T-norm of inputs
            products = np.product(fuzzified, axis=1)

            alphas.append(products)

            # 3. layer: Normalization
            normalized = (products / (underflow_ctrl if np.sum(products)
                                      == 0 else np.sum(products)))

            x, y = x_y.T

            # 4. layer: Defuzzification (w * (p * x1 + q * x2 + r))
            deffuzified = normalized.dot(self.p * x + self.q * y + self.r)

            outs.append(deffuzified)

        return np.array(outs), np.array(alphas)

    def train(self, X_Y, z, learning_rate=1e-3, learning_rate_fuzzification=1e-3, lr_decay=0.9, decay_interval=1000, n_epochs=10000, algorithm='backprop', batch_size=10, underflow_ctrl=1e-12, shuffle=True):

        if algorithm not in ['backprop', 'stohastic', 'minibatch']:
            raise ValueError('Algorithm not recognised')

        if algorithm == 'backprop':
            batch_size = X_Y.shape[0]

        elif algorithm == 'stohastic':
            batch_size = 1

        for i in range(n_epochs):

            if i % 100 == 0:
                preds = self.predict(X_Y, underflow_ctrl=underflow_ctrl)
                print('Iteration {}, error {}'.format(
                    i, self.mean_squared_err(z, preds)))

            if shuffle:
                indices = np.random.permutation(X_Y.shape[0])
            else:
                indices = np.arange(X_Y.shape[0])

            for j in range(0, X_Y.shape[0], batch_size):

                sliced_indices = indices[j:j + batch_size]

                preds, alphas = self.forward(
                    X_Y[sliced_indices], underflow_ctrl=underflow_ctrl)

                x, y = X_Y[sliced_indices].T

                x = x.reshape(x.shape[0], 1)
                y = y.reshape(y.shape[0], 1)

                dEk_dpreds = -1 * (z[sliced_indices] - preds)
                alpha_sum = np.sum(alphas, axis=1)

                dEk_dp = np.sum([[dEk_dpreds[k] * alphas[k][i] * x[k] / alpha_sum[k]
                                  for i in range(self.p.shape[0])] for k in range(len(x))], axis=0)

                dEk_dq = np.sum([[dEk_dpreds[k] * alphas[k][i] * y[k] / alpha_sum[k]
                                  for i in range(self.q.shape[0])] for k in range(len(y))], axis=0)

                dEk_dr = np.sum([[dEk_dpreds[k] * alphas[k][i] / alpha_sum[k]
                                  for i in range(self.r.shape[0])] for k in range(len(x))], axis=0)

                #dEk_da = ...
                #dEk_db = ...

                self.p = self.p - learning_rate * dEk_dp
                self.q = self.q - learning_rate * dEk_dq
                self.r = self.r - learning_rate * dEk_dr
                #self.a = self.a - learning_rate_fuzzification * dEk_da
                #self.b = self.b - learning_rate_fuzzification * dEk_db

            if i % decay_interval == 0 and i != 0:
                learning_rate *= lr_decay
                learning_rate_fuzzification *= lr_decay

    def predict(self, X_Y, underflow_ctrl=1e-12):
        preds, _ = self.forward(X_Y, underflow_ctrl=underflow_ctrl)
        return preds


def draw_func(funcs, no_rules):
    x = np.linspace(-4, 4, 30)
    y = np.linspace(-4, 4, 30)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
    for i, f in enumerate(funcs):
        j, k = [0 if i < 2 else 1, i if i < 2 else i - 2]
        if i != 0:
            Z = []
            for x_ in x:
                row = []
                for y_ in y:
                    row.append(f(np.array([[x_, y_]])))
                Z.append(row)
            Z = np.array(Z).reshape(30, 30)
            ax[j, k].set_title(
                'Predicted using {} rules'.format(no_rules[i - 1]))
        else:
            Z = func(X, Y)
            ax[j, k].set_title('Original function')

        ax[j, k].plot_surface(X, Y, Z, rstride=1, cstride=1,
                              cmap='viridis', edgecolor='none')
        ax[j, k].set_xlabel('X')
        ax[j, k].set_ylabel('Y')
        ax[j, k].set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    X_Y = []
    z = []

    def func(x, y): return ((x - 1) ** 2 + (y + 2) **
                            2 - 5 * x * y + 3) * np.cos(x / 5) ** 2

    for x in range(-4, 4):
        for y in range(-4, 4):
            X_Y.append([x, y])
            z.append([func(x, y)])

    X_Y = np.array(X_Y)
    z = np.array(z)

    anfis = ANFIS(no_rules=4)

    anfis4.train(X_Y, z, n_epochs=10000, learning_rate=1e-2,
                 lr_decay=1, decay_interval=1000, shuffle=True)

    anfis4 = ANFIS.load_model('model4.npz')

    # anfis1 = ANFIS(no_rules=1)
    # anfis1.train(X_Y, z, n_epochs=10000, learning_rate=1e-3,
    #              lr_decay=1, decay_interval=1000, shuffle=True)

    # anfis1.save_model('model1.npz')

    # anfis1 = ANFIS.load_model('model1.npz')

    # anfis2 = ANFIS(no_rules=2)
    # anfis2.train(X_Y, z, n_epochs=10000, learning_rate=5e-3,
    #              lr_decay=1, decay_interval=1000, shuffle=True)

    # anfis2.save_model('model2.npz')

    # anfis2 = ANFIS.load_model('model2.npz')

    # anfis4 = ANFIS(no_rules=4)
    # anfis4.train(X_Y, z, n_epochs=10000, learning_rate=1e-2,
    #              lr_decay=1, decay_interval=1000, shuffle=True)

    # anfis4 = ANFIS.load_model('model4.npz')

    # draw_func([func, anfis1.predict, anfis2.predict, anfis4.predict], no_rules=[1, 2, 4])
