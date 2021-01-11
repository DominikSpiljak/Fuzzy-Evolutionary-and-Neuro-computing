import numpy as np

class NeuralNet:

    def __init__(self, layers, dataset):
        self.layers = layers

        self.dataset = dataset

        no_weights = np.sum([self.layers[i + 2] * self.layers[i + 1]
                             for i in range(len(self.layers) - 2)])
        no_biases = np.sum(self.layers[2:])
        no_neuron_type1_weights = self.layers[1] * self.layers[0]
        no_neuron_type1_s = self.layers[1] * self.layers[0]

        self.no_params = no_weights + no_biases + no_neuron_type1_weights + no_neuron_type1_s
    
    def get_num_params(self):
        return self.no_params

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    def forward(self, params, X):
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            elif i == 1:
                for j in range(layer):
                    neuron_params = params[j * 4:j * 4 + 4]
                    layer_outputs.append(np.sqrt(np.sum(np.square(
                                    X - np.array(neuron_params[:2])), axis=1)) / np.sqrt(np.sum(np.square(np.array(neuron_params[2:])))))
                layer_outputs = np.array(layer_outputs)
                layer_outputs = layer_outputs.reshape(layer_outputs.shape[1], layer_outputs.shape[0])
                last_idx = (layer - 1) * 4 + 4
            else:
                weights = np.array(params[last_idx:last_idx + self.layers[i - 1] * self.layers[i]]).reshape(self.layers[i - 1], self.layers[i])
                last_idx = last_idx + self.layers[i - 1] * self.layers[i]
                biases = np.array(params[last_idx:last_idx + self.layers[i]])
                last_idx = last_idx + self.layers[i]

                layer_outputs = NeuralNet.sigmoid(layer_outputs.dot(weights) + biases)
        
        return layer_outputs


    def calculate_error(self, params):
        return np.mean(np.sum(np.square(self.forward(params, self.dataset.X) - self.dataset.y), axis=1))
                
    def predict(self, params, X):
        return np.argmax(self.forward(params, X), axis=1)