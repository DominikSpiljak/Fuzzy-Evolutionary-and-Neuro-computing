import numpy as np
import matplotlib.pyplot as plt
import pickle
from genetic_algorithm import GeneticAlgorithm
import time
import warnings


class Dataset:
    def __init__(self, dataset):
        self.read_dataset(dataset)

    def read_dataset(self, dataset):
        X = []
        y = []
        with open(dataset, 'r') as inp:
            for line in inp:
                if len(line.strip()):
                    line = line.strip().split('\t')
                    X.append([float(line[0]), float(line[1])])
                    y.append(list(map(float, line[2:])))
        self.X = np.array(X)
        self.y = np.array(y)

    def get_sample(self, i):
        return self.X[i], self.y[i]

    def size(self):
        return len(self.X)


class EvolutionalNeuralNet:

    @staticmethod
    def get_no_params(layers):
        no_weights = np.sum([layers[i + 1] * layers[i + 2]
                             for i in range(len(layers) - 2)])
        no_biases = np.sum(layers[2:])
        no_neuron_type1_weights = layers[1] * layers[0]
        no_neuron_type1_s = layers[1] * layers[0]
        return no_weights + no_biases + no_neuron_type1_weights + no_neuron_type1_s

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-1 * X))

    @staticmethod
    def error(dataset, params, layers):
        return np.mean(np.sum(np.square(EvolutionalNeuralNet.forward(dataset.X, params, layers) - dataset.y), axis=1))

    @staticmethod
    def decode_params(params, layers):
        weights = []
        biases = []
        neuron_type1_weights = []
        neuron_type1_s = []
        index = 0
        # Decode weights
        for i in range(len(layers) - 2):
            weights.append(np.array(
                params[index:index + layers[i + 1] * layers[i + 2]]).reshape(layers[i + 1], layers[i + 2]))
            index += layers[i + 1] * layers[i + 2]

        # Decode biases
        for layer in layers[2:]:
            biases.append(np.array(params[index:index + layer]))
            index += layer

        # Decode neuron_type1_weights
        neuron_type1_weights = np.array(
            params[index:index + layers[1] * layers[0]]).reshape(layers[1], layers[0])

        index += layers[1] * layers[0]

        # Decode neuron_type1_s
        neuron_type1_s = np.array(
            params[index:index + layers[1] * layers[0]]).reshape(layers[1], layers[0])

        return np.array(weights), np.array(biases), np.array(neuron_type1_weights), np.array(neuron_type1_s)

    @staticmethod
    def encode_params(weights, biases, neuron_type1_weights, neuron_type1_s):
        weights_flat = []
        biases_flat = []
        for layer in weights:
            weights_flat.extend(layer.flatten())
        for layer in biases:
            biases_flat.extend(layer.flatten())

        return np.array([*weights_flat, *biases_flat, *list(neuron_type1_weights.flatten()), *list(neuron_type1_s.flatten())])

    @staticmethod
    def forward(X, params, layers, trace=False):
        weights, biases, neuron_type1_weights, neuron_type1_s = EvolutionalNeuralNet.decode_params(
            params, layers)
        outs = []
        trace_ = []
        for x in X:
            x_trace = []
            last_out = 1 / (1 + np.linalg.norm(neuron_type1_weights - x,
                                               axis=1) / np.linalg.norm(neuron_type1_s, axis=1))
            x_trace.append(last_out.reshape(len(last_out), 1))
            for i in range(len(layers[2:])):
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        last_out = EvolutionalNeuralNet.sigmoid(
                            last_out.dot(weights[i]) + biases[i])
                    except Warning as e:
                        print('Overflow: Weights: {}, Biases: {}'.format(
                            weights[i], biases[i]), e)
                x_trace.append(last_out.reshape(len(last_out), 1))
            outs.append(last_out)
            trace_.append(x_trace)

        if trace:
            return np.array(outs), np.array(trace_)
        else:
            return np.array(outs)

    @staticmethod
    def train(params, dataset, layers, n_iter, lr):
        # TODO: Finish updating neuron type 1 params
        for i in range(n_iter):
            """print('Iteration {}, error {}'.format(
                i, EvolutionalNeuralNet.error(dataset, params, layers)))"""
            preds, trace = EvolutionalNeuralNet.forward(
                dataset.X, params, layers, trace=True)

            weights, biases, neuron_type1_weights, neuron_type1_s = EvolutionalNeuralNet.decode_params(
                params, layers)

            for i, x_trace in enumerate(trace):
                bias_errs = []
                weight_errs = []
                x_trace_r = np.array(list(reversed(x_trace)))
                for j, layer_out in enumerate(x_trace_r):
                    if j == 0:
                        out_err = (preds[j] - dataset.y[j]
                                   ).reshape(len(preds[j]), 1)
                    else:
                        out_err = weights[-j].dot(in_err)

                    in_err = np.multiply(
                        out_err, np.multiply(layer_out, 1 - layer_out))
                    if j != len(x_trace) - 1:
                        weight_err = x_trace_r[j + 1].dot(in_err.T)
                        weight_errs.insert(0, weight_err)
                        bias_errs.insert(0, in_err)

                for j in range(len(x_trace) - 1):
                    weights[j] -= lr * weight_errs[j]
                    biases[j] -= lr * bias_errs[j].flatten()

            params = EvolutionalNeuralNet.encode_params(
                weights, biases, neuron_type1_weights, neuron_type1_s)

        return params, EvolutionalNeuralNet.error(dataset, params, layers)

    @ staticmethod
    def predict(X, params):
        return np.argmax(EvolutionalNeuralNet.forward(X, params, layers), axis=1)


class Individual:

    def __init__(self, value, dataset, layers):
        self.value, self.fitness = EvolutionalNeuralNet.train(
            value, ds, layers, n_iter=10, lr=1e-5)

    def __eq__(self, other):
        return np.all(self.value == other.value)

    def __str__(self):
        return "Fitness = {}".format(self.fitness)

    def save_individual(self, filename):
        with open(filename, 'wb') as out:
            pickle.dump(self, out)


def generate_population(population_size, no_params):
    def population_generation():
        population = []
        for _ in range(population_size):
            population.append(Individual(
                np.random.rand(no_params) * 2 - 1, ds, layers))

        return population

    return population_generation


def roulette_selection(elitism=True, no_elites=1):

    def selection(population):
        new_population = []
        comb_population = []

        sorted_population = sorted(
            population, key=lambda individual: individual.fitness)

        if elitism:
            new_population.extend(sorted_population[:no_elites])
            no_combs = len(population) - no_elites
        else:
            no_combs = len(population)

        fitnesses = np.array([ind.fitness for ind in population])
        probs = fitnesses / sum(fitnesses)

        comb_population = np.random.choice(
            population, p=probs, size=(no_combs, 2))

        return new_population, comb_population

    return selection


def weights_recombination_cross():
    def recombination_func(comb_population):
        children = []
        for pair in comb_population:
            indices = (np.random.rand(len(pair[0].value)) >= .5).astype('int')
            child_value = []

            for i in range(len(indices)):
                child_value.append(pair[indices[i]].value[i])

            children.append(Individual(child_value, ds, layers))

        return children

    return recombination_func


def neurons_recombination_cross():
    def recombination_func(comb_population):
        children = []
        for pair in comb_population:
            child_value = []
            pair_params = []
            pair_params.append(EvolutionalNeuralNet.decode_params(
                pair[0].value, layers))
            pair_params.append(EvolutionalNeuralNet.decode_params(
                pair[1].value, layers))

            child_neuron_type_1_indices = (
                np.random.rand(layers[1]) >= .5).astype('int')

            child_neuron_type1_weights = []
            child_neuron_type1_s = []

            for i in range(layers[1]):
                child_neuron_type1_weights.append(
                    pair_params[child_neuron_type_1_indices[i]][2][i])
                child_neuron_type1_s.append(
                    pair_params[child_neuron_type_1_indices[i]][3][i])

            child_neuron_weights = []
            child_neuron_biases = []
            for layer in layers[2:]:
                child_neuron_type_1_indices = (
                    np.random.rand(layer) >= .5).astype('int')
                child_neuron_weights_layer = []
                child_neuron_biases_layer = []

                for i in range(layer):
                    child_neuron_weights_layer.append(
                        pair_params[child_neuron_type_1_indices[i]][0].T[i])
                    child_neuron_biases_layer.append(
                        pair_params[child_neuron_type_1_indices[i]][1].T[i])

                child_neuron_weights.append(
                    np.array(child_neuron_weights_layer).T)
                child_neuron_biases.append(
                    np.array(child_neuron_biases_layer).T)

            child_value = EvolutionalNeuralNet.encode_params(np.array(child_neuron_weights), np.array(
                child_neuron_biases), np.array(child_neuron_type1_weights), np.array(child_neuron_type1_s))

            children.append(Individual(child_value, ds, layers))

        return children

    return recombination_func


def simulated_binary_cross():
    def simulated_binary_function(comb_population):
        children = []
        for pair in comb_population:
            rand = np.random.rand(pair[0].value.shape[0])
            child_vals = pair[0].value * rand + pair[1].value * (1 - rand)

            children.append(Individual(child_vals, ds, layers))
        return children

    return simulated_binary_function


def cross_chooser(cross_list):
    def cross(comb_population):
        cross_func = np.random.choice(cross_list)

        return cross_func(comb_population)

    return cross


def mutation_1(mutation_probabilty, deviation):
    def mutation_func(children):
        for individual in children:
            for j in range(len(individual.value)):
                rand = np.random.rand()
                if rand < mutation_probabilty:
                    rand_mutate = np.random.normal(0, deviation)
                    individual.value[j] += rand_mutate
        return children

    return mutation_func


def mutation_2(mutation_probabilty, deviation):
    def mutation_func(children):
        for individual in children:
            for j in range(len(individual.value)):
                rand = np.random.rand()
                if rand < mutation_probabilty:
                    rand_mutate = np.random.normal(0, deviation)
                    individual.value[j] = rand_mutate
        return children

    return mutation_func


def mutation_chooser(mutation_list, probs):
    probs = np.array(probs)
    probs = probs / np.sum(probs)

    def mutation_func(children):

        mutation = np.random.choice(mutation_list, p=probs)

        return mutation(children)

    return mutation_func


def solution():
    def solution_func(population):
        sorted_population = sorted(
            population, key=lambda individual: individual.fitness)
        return sorted_population[0]
    return solution_func


def plot_data(dataset, save_file=None, neuron_weights=None, model_specs=None):
    fig, axes = plt.subplots(1)
    fig.set_size_inches(20, 15)
    fig.suptitle('Vizualizacija podataka')
    if model_specs is not None:
        fig.suptitle('Vizualizacija podataka, model: ({})'.format(
            ", ".join([k + ": " + str(v) for k, v in model_specs.items()])))
    markers = [".", ",", "o"]
    colors = ['r', 'g', 'b']
    for i in range(dataset.size()):
        x, y = dataset.get_sample(i)
        label = np.argmax(y)
        axes.scatter(x[0], x[1], marker=markers[label], c=colors[label])

    if neuron_weights is not None:
        for weight in neuron_weights:
            axes.scatter(weight[0], weight[1], marker="*", c="k")

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
    global ds, layers
    ds = Dataset('dataset.txt')
    layers = [2, 8, 3]
    # test_neuron(2, [1, 0.25, 4], save_file='test_neuron.png')
    # plot_data(ds, save_file='data_visualisation.png')
    population_size = 100
    num_iter = 100000
    no_elites = 5
    mutation_chooser_probs = [70, 15, 15]
    mutation_prob = 0.1
    genetic_algorithm = GeneticAlgorithm(population_generation=generate_population(population_size, EvolutionalNeuralNet.get_no_params(layers)),
                                         num_iter=num_iter,
                                         selection=roulette_selection(
                                         elitism=True, no_elites=no_elites),
                                         combination=cross_chooser(
                                         [weights_recombination_cross(), neurons_recombination_cross(), simulated_binary_cross()]),
                                         mutation=mutation_chooser([mutation_1(mutation_prob, 0.3),
                                                                    mutation_1(
                                                                        mutation_prob, 1),
                                                                    mutation_2(mutation_prob, 0.3)],
                                                                   probs=mutation_chooser_probs),
                                         solution=solution(),
                                         goal_fitness=1e-7)

    start_time = time.time()
    best = genetic_algorithm.evolution()
    print("--- {} seconds ---".format(time.time() - start_time))

    best.save_individual("best_individual_{}_2.pickle".format(
        ''.join([str(layer) for layer in layers])))

    _, _, neuron_type1_weights, _ = EvolutionalNeuralNet.decode_params(
        best.value, layers)

    plot_data(ds, neuron_weights=neuron_type1_weights, model_specs={
        "pop_size": population_size,
        "num_iter": num_iter,
        "no_elites": no_elites,
        "mutation_chooser_probs": ', '.join([str(prob) for prob in mutation_chooser_probs]),
        "mutation_prob": mutation_prob,
        "layers": ', '.join([str(layer) for layer in layers])
    },
        save_file="data_visualisation_with_neuron_weights_{}_2.png".format(
        ''.join([str(layer) for layer in layers])))


if __name__ == "__main__":
    """ds = Dataset('dataset.txt')
    layers = [2, 8, 3]
    Individual(np.random.rand(
        EvolutionalNeuralNet.get_no_params(layers)) * 2 - 1, ds, layers)"""
    main()
