from individual import Individual
from genetic_algorithm import GeneticAlgorithm
import random
import math


def read_dataset(filepath):
    x = []
    y = []
    targets = []

    with open(filepath, 'r') as dataset:
        for line in dataset:
            line = line.strip().split('\t')
            line_x, line_y, line_target = line
            x.append(float(line_x))
            y.append(float(line_y))
            targets.append(float(line_target))
        return list(zip(x, y)), targets


def beta_function(b0, b1, b2, b3, b4, x_y):
    vals = []
    for entry in x_y:
        x, y = entry
        f1 = math.sin(b0 + b1 * x)
        f2 = b2 * math.cos(x * (b3 + y))
        f3 = 1 / (1 + math.exp((x - b4) ** 2))

        vals.append(f1 + f2 * f3)

    return vals


def mse(x_y, targets):
    def mean_squared_err(b0, b1, b2, b3, b4):
        vals = beta_function(b0, b1, b2, b3, b4, x_y)
        return 1 / len(targets) * sum([(val - target) ** 2 for val, target in zip(vals, targets)])
    return mean_squared_err


def generate_population(population_size, mse_func):
    def population_generation():
        population = []
        for _ in range(population_size):
            individual_vals = [random.random() * 8 - 4 for i in range(5)]
            population.append(Individual(
                individual_vals, mse_func))

        return population

    return population_generation


def generative_selection(elitism, no_elites):
    def choose_index(proportions):
        maxi = proportions[-1]
        rand = random.random() * maxi
        i = 0
        while proportions[i] < rand:
            i += 1
        return i - 1

    def selection(population):
        new_population = []
        comb_population = []
        if elitism:
            sorted_population = sorted(
                population, key=lambda individual: individual.fitness)

            new_population.extend(sorted_population[:no_elites])

        # Proportions calculation
        proportions = [1/population[0].fitness]
        for individual in population[1:]:
            proportions.append(proportions[-1] + 1/individual.fitness)

        for _ in range(len(population) - no_elites):
            comb_population.append(
                [population[choose_index(proportions)], population[choose_index(proportions)]])

        return new_population, comb_population

    return selection


def mean_cross(mse_func):
    def mean_cross_function(comb_population):
        children = []
        for pair in comb_population:
            individual0_vals = pair[0].value
            individual1_vals = pair[1].value
            child_vals = []
            for val1, val2 in zip(individual0_vals, individual1_vals):
                child_vals.append((val1 + val2) / 2)
            children.append(Individual(child_vals, mse_func))
        return children

    return mean_cross_function


def solution():
    def solution_func(population):
        sorted_population = sorted(
            population, key=lambda individual: individual.fitness)
        return sorted_population[0]
    return solution_func


def mutation(mutation_probabilty):
    def mutation_func(children):
        for individual in children:
            for j in range(len(individual.value)):
                rand = random.random()
                if rand < mutation_probabilty:
                    rand_mutate = random.random() * 8 - 4
                    individual.value[j] = rand_mutate
        return children

    return mutation_func


def main():
    x_y, targets = read_dataset('datasets/dataset1.txt')
    genetic = GeneticAlgorithm(population_generation=generate_population(population_size=400, mse_func=mse(x_y, targets)),
                               num_iter=500,
                               selection=generative_selection(
                                   elitism=True, no_elites=50),
                               combination=mean_cross(
                                   mse_func=mse(x_y, targets)),
                               mutation=mutation(0.01),
                               solution=solution())

    print(str(genetic.evolution()))


if __name__ == "__main__":
    main()
