class GeneticAlgorithm:

    def __init__(self, population_generation, num_iter, selection, combination, mutation, solution):
        self.population_generation = population_generation
        self.num_iter = num_iter
        self.selection = selection
        self.combination = combination
        self.mutation = mutation
        self.solution = solution

    def evolution(self):
        # Template method
        population = self.population_generation()
        min_fitness = self.solution(population).fitness

        for i in range(self.num_iter):
            population, comb_population = self.selection(population)
            population.extend(self.mutation(self.combination(comb_population)))

            if self.solution(population).fitness < min_fitness:
                min_fitness = self.solution(population).fitness
                print("Found new best, iteration = {}, {}".format(
                    i, self.solution(population)))

        return self.solution(population)
