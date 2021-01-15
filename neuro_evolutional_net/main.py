from neural_net import NeuralNet
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import genetics
from genetic_algorithm import GeneticAlgorithm
import time


def plot_data(dataset, save_file=None, neural_net=None, params=None, model_specs=None):
    fig, axes = plt.subplots(1)
    fig.set_size_inches(20, 15)
    fig.suptitle('Vizualizacija podataka')
    if model_specs is not None:
        fig.suptitle('Vizualizacija podataka, model: ({})'.format(
            ", ".join([k + ": " + str(v) for k, v in model_specs.items()])))
    markers = [".", ",", "o"]
    colors = ['r', 'g', 'b']
    if params is None:
        for i in range(dataset.size()):
            x, y = dataset.get_sample(i)
            label = np.argmax(y)
            axes.scatter(x[0], x[1], marker=markers[label], c=colors[label])

    else:
        preds = neural_net.predict(neural_net.decode_params(params), dataset.X)

        for i in range(dataset.size()):
            x = dataset.X[i]
            if preds[i] == np.argmax(dataset.y[i]):
                axes.scatter(x[0], x[1], marker=markers[2], c=colors[2])
            else:
                axes.scatter(x[0], x[1], marker=markers[0], c=colors[0])

        w_type_1, s_type_1, _, _ = neural_net.decode_params(params)
        for i in range(int(model_specs['layers'].split(',')[1])):
            axes.scatter(w_type_1[:, 0], w_type_1[:, 1],
                         marker="*", c=colors[1])
        print('s = {}'.format(s_type_1))

    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def test_neuron(w, s, save_file=None):
    fig, axes = plt.subplots(1)
    fig.set_size_inches(18, 10)
    fig.suptitle('Ovisnost izlaza neurona o parametru s')
    for s_ in sorted(s):
        x = np.linspace(-8, 10, num=100)
        y = 1 / (1 + np.abs(x - w) / s_)

        axes.plot(x, y, label='s = {}'.format(s_))
        axes.legend()

    if save_file is not None:
        plt.savefig(save_file)
    plt.show()


def main():
    dataset = Dataset('dataset.txt')
    layers = [2, 8, 3]
    neural_net = NeuralNet(layers, dataset)
    #test_neuron(2, [1, 0.25, 4], save_file='test_neuron.png')
    #plot_data(dataset, save_file='data_visualisation.png')
    population_size = 40
    num_iter = 5000000
    k = 3
    mutation_chooser_probs = [10, 3, 5]
    mutation_prob = 0.05
    genetic_algorithm = GeneticAlgorithm(population_generation=genetics.generate_population(neural_net.get_num_params(), population_size, neural_net),
                                         num_iter=num_iter,
                                         selection=genetics.tournament_selection(
        population_size, k=k),
        combination=genetics.cross_chooser(
        [genetics.weight_recombination(neural_net),
            genetics.simulated_binary_recombination(
            neural_net),
            genetics.whole_arithmetic_recombination(neural_net)]),
        mutation=genetics.mutation_chooser([genetics.mutation_1(mutation_prob, 0.05),
                                            genetics.mutation_1(
            mutation_prob, 0.3),
            genetics.mutation_2(mutation_prob, 1)],
        probs=mutation_chooser_probs, neural_net=neural_net),
        solution=genetics.solution(),
        goal_error=1e-7)

    start_time = time.time()
    best = genetic_algorithm.evolution(neural_net)
    print("--- {} seconds ---".format(time.time() - start_time))

    best.save_individual("best_individual_{}_1.pickle".format(
        ''.join([str(layer) for layer in layers])))

    print(neural_net.calculate_error(
        [best.w_type_1, best.s_type_1, best.w, best.b]))

    plot_data(dataset, neural_net=neural_net, model_specs={
        "pop_size": population_size,
        "num_iter": num_iter,
        "k": k,
        "mutation_chooser_probs": ', '.join([str(prob) for prob in mutation_chooser_probs]),
        "mutation_prob": mutation_prob,
        "layers": ', '.join([str(layer) for layer in layers])
    },
        params=best.value,
        save_file="data_visualisation_with_neuron_weights_{}_1.png".format(
        ''.join([str(layer) for layer in layers])))


if __name__ == "__main__":
    # main()
    import pickle
    dataset = Dataset('dataset.txt')
    layers = [2, 8, 3]
    neural_net = NeuralNet(layers, dataset)
    with open('best_individual_283.pickle', 'rb') as inp:
        best = pickle.load(inp)

    # neural_net.show(best.value, save_file='neural_net_283.png')

    w_type_1, s_type_1, w, b = neural_net.decode_params(best.value)

    print(s_type_1)
