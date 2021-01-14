import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, layers, dataset):
        self.layers = layers

        self.dataset = dataset

        no_weights = np.sum([self.layers[i + 2] * self.layers[i + 1]
                             for i in range(len(self.layers) - 2)])
        no_biases = np.sum(self.layers[2:])
        no_neuron_type1_weights = self.layers[1] * self.layers[0]
        no_neuron_type1_s = self.layers[1] * self.layers[0]

        self.no_params = no_weights + no_biases + \
            no_neuron_type1_weights + no_neuron_type1_s

    def get_num_params(self):
        return self.no_params

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def forward(self, w_type_1, s_type_1, w, b, X):
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            elif i == 1:
                layer_outputs = np.array(
                    [1 / (1 + np.sqrt(np.sum(np.square((X - w_type_1[j]) / s_type_1[j]), axis=1))) for j in range(layer)]).T

            else:
                weights = w[i - 2]
                biases = b[i - 2]

                layer_outputs = NeuralNet.sigmoid(
                    layer_outputs.dot(weights) + biases)

        return layer_outputs

    def calculate_error(self, params):
        w_type_1, s_type_1, w, b = params
        return np.mean(np.sum(np.square(self.forward(w_type_1, s_type_1, w, b, self.dataset.X) - self.dataset.y), axis=1))

    def predict(self, params, X):
        w_type_1, s_type_1, w, b = params
        return np.argmax(self.forward(w_type_1, s_type_1, w, b, X), axis=1)

    def decode_params(self, params):
        w_type_1 = []
        s_type_1 = []

        weights = []
        biases = []

        last_idx = 0

        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            if i == 1:
                for _ in range(layer):
                    w_type_1.append(params[last_idx: last_idx + 2])
                    s_type_1.append(params[last_idx + 2: last_idx + 4])
                    last_idx = last_idx + 4
            else:
                w = np.array(params[last_idx:last_idx + self.layers[i - 1] *
                                    self.layers[i]]).reshape(self.layers[i - 1], self.layers[i])
                weights.append(w)
                last_idx = last_idx + self.layers[i - 1] * self.layers[i]
                b = np.array(params[last_idx:last_idx + self.layers[i]])
                biases.append(b)
                last_idx = last_idx + self.layers[i]

        return np.array(w_type_1), np.array(s_type_1), np.array(weights), np.array(biases)

    def show(self, params, save_file=None):

        ax = plt.figure(figsize=(12, 12)).gca()
        left = .1
        right = .9
        bottom = .1
        top = .9
        layer_sizes = self.layers
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)

        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                    ax.add_artist(line)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
        if save_file is not None:
            plt.savefig(save_file)

        plt.show()
