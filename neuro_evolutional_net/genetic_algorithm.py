from tqdm import tqdm
from individual import Individual

class GeneticAlgorithm:

    def __init__(self, population_generation, num_iter, selection, combination, mutation, solution, goal_error):
        self.population_generation = population_generation
        self.num_iter = num_iter
        self.selection = selection
        self.combination = combination
        self.mutation = mutation
        self.solution = solution
        self.goal_error = goal_error

    def evolution(self, neural_net):
        # Template method
        population = self.population_generation()
        min_error = self.solution(population).error

        for i in tqdm(range(self.num_iter)):
            population, comb_population = self.selection(population)
            combined_population = self.combination(comb_population)
            mutated_population = self.mutation(combined_population)
            population.extend(mutated_population)

            iteration_min_error = self.solution(population).error

            if iteration_min_error < min_error:
                min_error = iteration_min_error
                print("Found new best, iteration = {}, {}".format(
                    i, self.solution(population)))
                if iteration_min_error < self.goal_error:
                    print("Reached goal error, terminating.")
                    break
        
        return self.solution(population)