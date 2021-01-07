import numpy as np
import matplotlib.pyplot as plt
import json
from genetic_algorithm import GeneticAlgorithm


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
    def __init__(self, input_layer_size=None, hidden_layer_sizes=None, output_layer_size=None):
        if input_layer_size is not None:
            self.layers = [input_layer_size, *
                           hidden_layer_sizes, output_layer_size]

            weights = np.array([np.random.randn(
                self.layers[i + 1], self.layers[i + 2]) for i in range(len(self.layers) - 2)])
            biases = np.array([np.random.randn((layer))
                               for layer in self.layers[2:]])
            neuron_type1_weights = np.random.rand(
                self.layers[1], self.layers[0])
            neuron_type1_s = np.random.randn(self.layers[1], self.layers[0])

            weights_list_flattened = [list(layer.flatten())
                                      for layer in weights]
            weights_list = []
            for layer in weights_list_flattened:
                weights_list.extend(layer)
            biases_list_flattened = [list(layer.flatten()) for layer in biases]
            biases_list = []
            for layer in biases_list_flattened:
                biases_list.extend(layer)
            neuron_type1_weights_list = list(neuron_type1_weights.flatten())
            neuron_type1_s_list = list(neuron_type1_s.flatten())
            self.params = [*weights_list, *biases_list,
                           *neuron_type1_weights_list, *neuron_type1_s_list]

    @staticmethod
    def load_model(path):
        with open(path, 'r') as inp:
            state_dict = json.load(inp)

        model = EvolutionalNeuralNet()
        model.layers = state_dict['layers']
        model.params = state_dict['params']
        return model

    def get_no_params(self):
        return len(self.params)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-1 * X))

    def error(self, dataset):
        return 1 / dataset.size() * (np.sum(np.square(self.forward(dataset.X) - dataset.y)))

    def save(self, path):
        state_dict = {'layers': self.layers, 'params': self.params}
        with open(path, 'w') as out:
            json.dump(state_dict, out)

    def decode_params(self):
        weights = []
        biases = []
        neuron_type1_weights = []
        neuron_type1_s = []
        index = 0
        # Decode weights
        for i in range(len(self.layers) - 2):
            weights.append(np.array(
                self.params[index:index + self.layers[i + 1] * self.layers[i + 2]]).reshape(self.layers[i + 1], self.layers[i + 2]))
            index += self.layers[i + 1] * self.layers[i + 2]

        # Decode biases
        for layer in self.layers[2:]:
            biases.append(np.array(self.params[index:index + layer]))
            index += layer

        # Decode neuron_type1_weights
        neuron_type1_weights = np.array(
            self.params[index:index + self.layers[1] * self.layers[0]]).reshape(self.layers[1], self.layers[0])

        index += self.layers[1] * self.layers[0]

        # Decode neuron_type1_s
        neuron_type1_s = np.array(
            self.params[index:index + self.layers[1] * self.layers[0]]).reshape(self.layers[1], self.layers[0])

        return np.array(weights), np.array(biases), np.array(neuron_type1_weights), np.array(neuron_type1_s)

    def forward(self, X):
        weights, biases, neuron_type1_weights, neuron_type1_s = self.decode_params()
        outs = []
        for x in X:
            last_out = 1 / \
                (1 + np.array([np.linalg.norm(neuron_type1_weights[i] - x) / np.linalg.norm(
                    neuron_type1_s[i]) for i in range(len(neuron_type1_weights))]))
            for i in range(len(self.layers[2:])):
                last_out = self.sigmoid(last_out.dot(weights[i]) + biases[i])
            outs.append(last_out)
        return np.array(outs)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def get_param_list(self):
        return self.params


class Individual:

    def __init__(self, neural_net, dataset):
        self.neural_net = neural_net
        self.value = np.array(neural_net.get_param_list())
        self.fitness = self.neural_net.error(dataset)

    @staticmethod
    def create_with_value(value, layers, dataset):
        ind = Individual(EvolutionalNeuralNet(*layers), dataset)
        return ind

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return "Fitness = {}".format(self.fitness)


def generate_population(population_size):
    def population_generation():
        population = []
        for _ in range(population_size):
            population.append(Individual(
                EvolutionalNeuralNet(*layers), ds))

        return population

    return population_generation


def roulette_selection(elitism=True, no_elites=1):
    def choose_index(proportions):
        maxi = proportions[-1]
        rand = np.random.rand() * maxi
        i = 0
        while proportions[i] < rand:
            i += 1
        return i - 1

    def get_mapping(value, worst, best, lower_bound, upper_bound):
        return lower_bound + (upper_bound - lower_bound) * ((value - worst) / (best - worst))

    def selection(population):
        new_population = []
        comb_population = []

        sorted_population = sorted(
            population, key=lambda individual: individual.fitness)

        min_fitness = sorted_population[0].fitness
        max_fitness = sorted_population[-1].fitness

        if elitism:
            new_population.extend(sorted_population[:no_elites])
            no_combs = len(population) - no_elites
        else:
            no_combs = len(population)

        # Proportions calculation
        proportions = [get_mapping(
            population[0].fitness, max_fitness, min_fitness, 0, 1)]
        for individual in population[1:]:
            proportions.append(
                proportions[-1] + get_mapping(individual.fitness, max_fitness, min_fitness, 0, 1))

        for _ in range(no_combs):
            comb_population.append(
                [population[choose_index(proportions)], population[choose_index(proportions)]])

        return new_population, comb_population

    return selection


def mean_cross():
    def mean_cross_function(comb_population):
        children = []
        for pair in comb_population:
            individual0_vals = pair[0].value
            individual1_vals = pair[1].value
            child_vals = (individual0_vals + individual1_vals) / 2
            children.append(Individual.create_with_value(
                np.array(child_vals), layers, ds))
        return children

    return mean_cross_function


def arithmetic_cross_float():
    def arithmetic_cross_function(comb_population):
        children = []
        for pair in comb_population:
            rand = np.random.rand(pair[0].value.shape[0])
            child_vals = pair[0].value * rand + pair[1].value * (1 - rand)

            children.append(Individual.create_with_value(
                np.array(child_vals), layers, ds))
        return children

    return arithmetic_cross_function


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
    mutation_probs = []
    for i, prob in enumerate(probs):
        if i == 0:
            mutation_probs.append(prob)
        else:
            mutation_probs.append(mutation_probs[-1] + prob)

    def mutation_func(children):
        def choose_index(proportions):
            maxi = proportions[-1]
            rand = np.random.rand() * maxi
            i = 0
            while proportions[i] < rand:
                i += 1
            return i - 1

        mutation = mutation_list[choose_index(mutation_probs)]

        return mutation(children)

    return mutation_func


def solution():
    def solution_func(population):
        sorted_population = sorted(
            population, key=lambda individual: individual.fitness)
        return sorted_population[0]
    return solution_func


def plot_data(dataset, save_file=None, neuron_weights=None):
    fig, axes = plt.subplots(1)
    fig.set_size_inches(20, 15)

    fig.suptitle('Vizualizacija podataka')
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
    layers = [2, [8], 3]
    # test_neuron(2, [1, 0.25, 4], save_file='test_neuron.png')
    # plot_data(ds, save_file='data_visualisation.png')
    mutation_prob = 0.02
    genetic_algorithm = GeneticAlgorithm(population_generation=generate_population(100),
                                         num_iter=10000,
                                         selection=roulette_selection(
                                             elitism=True, no_elites=30),
                                         combination=cross_chooser(
                                             [arithmetic_cross_float(), mean_cross()]),
                                         mutation=mutation_chooser([mutation_1(mutation_prob, 1),
                                                                    mutation_1(
                                                                        mutation_prob, 5),
                                                                    mutation_2(mutation_prob, 1)],
                                                                   probs=[0.6, 0.2, 0.2]),
                                         solution=solution())

    best = genetic_algorithm.evolution()
    _, _, neuron_type1_weights, neuron_type1_s = best.neural_net.decode_params()
    plot_data(ds, neuron_weights=neuron_type1_weights)


if __name__ == "__main__":
    main()
