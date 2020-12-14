import numpy as np
import pickle


class ANFIS():

    def __init__(self, no_variables=None, no_rules=None):
        if no_variables is not None and no_rules is not None:
            self.pq = np.random.randn(no_rules, no_variables)
            self.r = np.zeros((no_rules, 1))

            self.a = np.random.randn(no_rules, no_variables)
            self.b = np.random.randn(no_rules, no_variables)

    def fuzzification_sigmoid(self, X, a, b):
        return 1 / (1 + np.exp(b * (X - a)))

    def forward(self, X, a, b, pq, r):
        # 1. layer: Fuzzification
        fuzzified = self.fuzzification_sigmoid(X, a, b)

        # 2. layer: Product of inputs
        products = np.product(fuzzified, axis=1)

        # 3. layer: Normalization
        normalized = products / sum(products)

        # 4. layer: Defuzzification (w * (p * x1 + q * x2 + r))
        deffuzified = normalized.dot(
            np.sum(np.multiply(pq, X), axis=1) + r)

        # 5. layer: Output layer (sum(wi * fi) / sum(w))
        out = sum(deffuzified) / sum(normalized)

        return out

    def train(self, X, y, learning_rate=1e-3, n_epochs=10000, algorithm='backprop', batch_size=10, shuffle=True):

        if algorithm not in ['backprop', 'stohastic', 'minibatch']:
            raise ValueError('Algorithm not recognised')

        if algorithm == 'backprop':
            batch_size = X.shape[0]

        elif algorithm == 'stohastic':
            batch_size = 1

        for i in range(n_epochs):

            if shuffle:
                indices = np.random.permutation(X.shape[0])
            else:
                indices = np.arange(X.shape[0])

            for j in range(0, X.shape[0], batch_size):
                weight_errs = []

                indices = indices[j:j + batch_size]

                preds = self.forward(
                    X[indices], self.a[indices], self.b[indices], self.pq[indices], self.r[indices])

    def predict(self, X):
        return self.forward(X, self.a, self.b, self.pq, self.r)


if __name__ == "__main__":
    anfis = ANFIS(2, 2)
    x = [[0, 0], [0, 0]]
    print(anfis.predict(np.array(x)))
