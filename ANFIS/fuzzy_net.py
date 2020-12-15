import numpy as np
import pickle


class ANFIS():

    def __init__(self, no_variables=2, no_rules=None):
        if no_variables is not None and no_rules is not None:
            self.p = np.random.randn(no_rules, 1)
            self.q = np.random.randn(no_rules, 1)
            self.r = np.zeros((no_rules, 1))

            self.a = np.random.randn(no_rules, no_variables)
            self.b = np.random.randn(no_rules, no_variables)

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

            # 2. layer: Product of inputs
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

    def train(self, X_Y, z, learning_rate=1e-3, n_epochs=10000, algorithm='backprop', batch_size=10, underflow_ctrl=1e-12, shuffle=True):

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
                weight_errs = []

                sliced_indices = indices[j:j + batch_size]

                preds, alphas = self.forward(
                    X_Y[sliced_indices], underflow_ctrl=underflow_ctrl)

            x, y = X_Y[sliced_indices].T

            x = x.reshape(x.shape[0], 1)
            y = y.reshape(y.shape[0], 1)

            Ek_pi = - 1 * (z - preds) * \
                ((x.T.dot(alphas)) / np.sum(alphas, axis=0))

            Ek_qi = - 1 * (z - preds) * \
                ((y.T.dot(alphas)) / np.sum(alphas, axis=0))

            Ek_ri = - 1 * (z - preds) * \
                ((alphas) / np.sum(alphas, axis=0))

            # TODO: fix dimensions

    def predict(self, X_Y, underflow_ctrl=1e-12):
        preds, _ = self.forward(X_Y, underflow_ctrl=underflow_ctrl)
        return preds


if __name__ == "__main__":
    anfis = ANFIS(2, 2)
    X_Y = [[0, 0], [1, 1], [2, 2], [3, 3]]
    z = [[0], [2], [4], [4]]
    anfis.train(np.array(X_Y), np.array(z), n_epochs=1, shuffle=False)
