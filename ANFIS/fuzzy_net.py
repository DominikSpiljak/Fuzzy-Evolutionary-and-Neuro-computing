import numpy as np
import pickle


class ANFIS():

    def __init__(self, no_variables=None, no_rules=None):
        if no_variables is not None and no_rules is not None:
            self.pq = np.random.randn(no_rules, no_variables)
            self.r = np.zeros((no_rules, 1))

            self.a = np.random.randn(no_rules, no_variables)
            self.b = np.random.randn(no_rules, no_variables)

    def fuzzification_sigmoid(self, X):
        return 1 / (1 + np.exp(self.b * (X - self.a)))

    def mean_squared_err(self, y_true, preds):
        return np.mean(np.square(y_true - preds))

    def forward(self, x, underflow_ctrl=1e-12):

        # 1. layer: Fuzzification
        fuzzified = self.fuzzification_sigmoid(x)

        # 2. layer: Product of inputs
        products = np.product(fuzzified, axis=1)

        # 3. layer: Normalization
        normalized = (products / (underflow_ctrl if np.sum(products)
                                  == 0 else np.sum(products)))

        # 4. layer: Defuzzification (w * (p * x1 + q * x2 + r))
        deffuzified = normalized.dot(
            np.sum(np.multiply(self.pq, x), axis=1) + self.r)

        # 5. layer: Output layer (sum(wi * fi) / sum(w))
        out = (np.sum(deffuzified) / (underflow_ctrl if np.sum(normalized)
                                      == 0 else np.sum(normalized)))

        return np.array([out]), np.array(normalized)

    def train(self, X, y, learning_rate=1e-3, n_epochs=10000, algorithm='backprop', underflow_ctrl=1e-12, shuffle=True):

        if algorithm not in ['backprop', 'stohastic']:
            raise ValueError('Algorithm not recognised')

        if algorithm == 'backprop':
            batch_size = X.shape[0]

        elif algorithm == 'stohastic':
            batch_size = 1

        for i in range(n_epochs):

            if i % 100 == 0:
                preds = self.predict(X, underflow_ctrl=underflow_ctrl)
                print('Iteration {}, error {}'.format(
                    i, self.mean_squared_err(y, preds)))

            if shuffle:
                indices = np.random.permutation(X.shape[0])
            else:
                indices = np.arange(X.shape[0])

            for j in range(0, X.shape[0], batch_size):
                sliced_indices = indices[j:j + batch_size]

                X_batch = X[sliced_indices]

                batch_preds = []
                batch_alphas = []

                for x in X_batch:
                    preds, alphas = self.forward(
                        x, underflow_ctrl=underflow_ctrl)
                    batch_preds.append(preds)
                    batch_alphas.append(alphas)

                batch_preds = np.array()

                dEk_dpi = -()

    def predict(self, X, underflow_ctrl=1e-12):
        all_preds = []
        for x in X:
            preds, _ = self.forward(x, underflow_ctrl=underflow_ctrl)
            all_preds.append(preds)
        return np.array(all_preds)


if __name__ == "__main__":
    anfis = ANFIS(2, 2)
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([[0], [2], [4], [6]])
    anfis.train(X, y, algorithm='backprop', n_epochs=1)
