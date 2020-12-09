import numpy as np
import pickle
from canvas import DrawingCanvas


class NeuralNet():

    def __init__(self, input_layer_size=None, hidden_layer_sizes=None, output_layer_size=None):
        if input_layer_size is not None and hidden_layer_sizes is not None and output_layer_size is not None:
            self.layers = [input_layer_size, *
                           hidden_layer_sizes, output_layer_size]
            self.weights = np.array([np.random.randn(
                self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as inp:
            weights = pickle.load(inp)

        model = NeuralNet()
        model.weights = weights
        return model

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-1 * X))

    def forward(self, X):
        self.forward_matrixes = [X]
        last_out = X
        for i in range(len(self.weights)):
            forward = self.sigmoid(
                last_out.dot(self.weights[i]))
            last_out = forward[:]
            self.forward_matrixes.insert(0, last_out)
        return last_out

    def mean_squared_err(self, y_true, preds):
        return np.mean(np.square(y_true - preds))

    def train(self, X, y, learning_rate=1e-3, n_epochs=10000, batch_size=None):

        for i in range(n_epochs):

            preds = self.forward(X)

            if i % 100 == 0:
                print('Epoch: {}, Error: {}'.format(
                    i, self.mean_squared_err(preds, y)))

            weight_errs = []

            for i in range(len(self.forward_matrixes[:-1])):
                if i == 0:
                    out_err = preds - y
                else:
                    out_err = in_err.dot(self.weights[-i].T)

                in_err = np.multiply(out_err, np.multiply(
                    self.forward_matrixes[i], 1 - self.forward_matrixes[i]))
                weight_err = in_err.T.dot(self.forward_matrixes[i + 1])
                weight_errs.insert(0, weight_err.T)

            for i in range(len(self.layers) - 1):
                self.weights[i] -= learning_rate * weight_errs[i]

    def predict(self, X):
        return self.forward(X)


def main():
    npzdata = np.load('dataset.npz')
    X = npzdata['X']
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = npzdata['y']

    #nn = NeuralNet.load_model('model.pickle')
    nn = NeuralNet(input_layer_size=X.shape[1],
                   hidden_layer_sizes=[16, 32, 16], output_layer_size=y.shape[1])
    nn.train(X, y, n_epochs=100000)
    with open('model.pickle', 'wb') as out:
        pickle.dump(nn.weights, out)

    DrawingCanvas(train_mode=False, model=nn)


if __name__ == "__main__":
    main()
