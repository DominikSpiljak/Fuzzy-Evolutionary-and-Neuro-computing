from tqdm import tqdm


class GeneticAlgorithm:

    def __init__(self, population_generation, num_iter, selection, combination, mutation, solution, goal_fitness):
        self.population_generation = population_generation
        self.num_iter = num_iter
        self.selection = selection
        self.combination = combination
        self.mutation = mutation
        self.solution = solution
        self.goal_fitness = goal_fitness

    def evolution(self):
        # Template method
        population = self.population_generation()
        min_fitness = self.solution(population).fitness

        for i in tqdm(range(self.num_iter)):
            population, comb_population = self.selection(population)
            population.extend(self.mutation(self.combination(comb_population)))

            iteration_min_fitness = self.solution(population).fitness

            if iteration_min_fitness < min_fitness:
                min_fitness = iteration_min_fitness
                print("Found new best, iteration = {}, {}".format(
                    i, self.solution(population)))
                if iteration_min_fitness < self.goal_fitness:
                    print("Reached goal fitness, terminating.")

        return self.solution(population)
