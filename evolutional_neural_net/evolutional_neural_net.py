import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import FormatStrFormatter
from neural_net import NeuralNet


class EvolutionalNeuralNet(NeuralNet):
    def __init__(self, input_layer_size=None, hidden_layer_sizes=None, output_layer_size=None):
        super(EvolutionalNeuralNet, self).__init__(
            input_layer_size, hidden_layer_sizes, output_layer_size)
        self.params = {}
        self.params["weights"] = self.weights
        self.params["neuron_type2_weights"] = np.array(
            [np.random.rand(layer, 1) for layer in [*hidden_layer_sizes[1:], output_layer_size]])
        self.params["neuron_type2_s"] = np.array(
            [np.random.randn(layer, 1) for layer in [*hidden_layer_sizes[1:], output_layer_size]])

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as inp:
            params = pickle.load(inp)

        model = EvolutionalNeuralNet()
        model.params = params
        return model

    def get_no_params(self):
        no_params = 0
        for weight_layer in self.params["weights"]:
            no_params += weight_layer.shape[0] * weight_layer.shape[1]
        for neuron_weight_layer in self.params["neuron_type2_weights"]:
            no_params += neuron_weight_layer.shape[0] * \
                neuron_weight_layer.shape[1]
        for neuron_s_layer in self.params["neuron_type2_s"]:
            no_params += neuron_s_layer.shape[0] * neuron_s_layer.shape[1]
        return no_params

    def save(self, path):
        with open(path, 'wb') as out:
            pickle.dump(self.params, out)

    def forward(self, X):
        pass

    def train(self, X, y, learning_rate=1e-3, n_epochs=10000, algorithm='backprop', batch_size=10, shuffle=True):
        pass


def read_dataset(dataset):
    X = []
    y = []
    with open(dataset, 'r') as inp:
        for line in inp:
            if len(line.strip()):
                line = line.strip().split('\t')
                X.append([float(line[0]), float(line[1])])
                y.append(list(map(float, line[2:])))
    return np.array(X), np.array(y)


def plot_data(X, y, save_file=None):
    fig, axes = plt.subplots(1)
    fig.set_size_inches(20, 15)
    fig.suptitle('Vizualizacija podataka')
    markers = [".", ",", "o"]
    colors = ['r', 'g', 'b']
    labels = np.argmax(y, axis=1)
    for x, label in zip(X, labels):
        axes.scatter(x[0], x[1], marker=markers[label], c=colors[label])

    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def test_neuron(w, s, save_file=None):
    fig, axes = plt.subplots(1)
    fig.set_size_inches(18, 10)
    fig.suptitle('Ovisnost izlaza neurona o parametru s')
    for s_ in sorted(s):
        x = np.linspace(-8, 10, num=100)
        y = 1 / (1 + (np.abs(x - w) / np.abs(s_)))

        axes.plot(x, y, label='s = {}'.format(s_))
        axes.legend()

    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def main():
    X, y = read_dataset('dataset.txt')
    #test_neuron(2, [1, 0.25, 4], save_file='test_neuron.png')
    #plot_data(X, y, save_file='data_visualisation.png')
    enn = EvolutionalNeuralNet(2, [8], 3)
    print(enn.get_no_params())


if __name__ == "__main__":
    main()
