import numpy as np
from individual import Individual


def generate_population(no_params, population_size, neural_net):
    def generator():
        return [Individual(np.random.rand(no_params) * 2 - 1, neural_net) for _ in range(population_size)]
    return generator


def tournament_selection(population_size, k=3):

    def selector(population):
        comb_population = []
        selected = []
        for _ in range(k):
            selected.append(population[np.random.randint(0, population_size)])
        selected_sorted = sorted(selected, key=lambda x: x.error)
        comb_population.append([selected_sorted[0], selected_sorted[1]])
        population.remove(selected_sorted[k-1])
        return population, comb_population

    return selector


def weight_recombination(neural_net):
    def recombinator(comb_population):
        children = []
        for parents in comb_population:
            child_vals = []
            rand = (np.random.rand(neural_net.get_num_params())
                    >= 0.5).astype('int')
            for i in range(neural_net.get_num_params()):
                child_vals.append(parents[rand[i]].get_value()[i])
            children.append(Individual(child_vals, neural_net))
        return children
    return recombinator


def simulated_binary_recombination(neural_net):
    def recombinator(comb_population):
        children = []
        for parents in comb_population:
            rand = np.random.rand(neural_net.get_num_params())
            child_vals = (1 - rand) * \
                parents[0].get_value() + rand * parents[1].get_value()
            children.append(Individual(child_vals, neural_net))
        return children
    return recombinator


def whole_arithmetic_recombination(neural_net):
    def recombinator(comb_population):
        children = []
        for parents in comb_population:
            child_vals = (parents[0].get_value() + parents[1].get_value()) / 2
            children.append(Individual(child_vals, neural_net))
        return children
    return recombinator


def cross_chooser(cross_list):
    def cross(comb_population):
        cross_func = np.random.choice(cross_list)
        return cross_func(comb_population)
    return cross


def mutation_1(mutation_prob, deviation, val_range=None):
    def mutator(children):
        for child in children:
            rand = np.random.rand(len(child.get_value()))
            for i in range(len(child.get_value())):
                if rand[i] < mutation_prob:
                    child.value[i] += np.random.normal(0, scale=deviation)
                    if val_range is not None:
                        if child.value[i] < val_range[0]:
                            child.value[i] = val_range[0]
                        elif child.value[i] > val_range[1]:
                            child.value[i] = val_range[1]
        return children
    return mutator


def mutation_2(mutation_prob, deviation, val_range=None):
    def mutator(children):
        for child in children:
            rand = np.random.rand(len(child.get_value()))
            for i in range(len(child.get_value())):
                if rand[i] < mutation_prob:
                    child.value[i] = np.random.normal(0, scale=deviation)
                    if val_range is not None:
                        if child.value[i] < val_range[0]:
                            child.value[i] = val_range[0]
                        elif child.value[i] > val_range[1]:
                            child.value[i] = val_range[1]
        return children
    return mutator


def mutation_chooser(mutation_list, probs, neural_net):
    probs = np.array(probs)
    probs = probs / np.sum(probs)

    def mutation_func(children):
        mutation = np.random.choice(mutation_list, p=probs)
        mutated = mutation(children)
        for c in mutated:
            c.error = neural_net.calculate_error(
                [c.w_type_1, c.s_type_1, c.w, c.b])

        return mutated

    return mutation_func


def solution():
    def solution_func(population):
        sorted_population = sorted(
            population, key=lambda individual: individual.error)
        return sorted_population[0]
    return solution_func
