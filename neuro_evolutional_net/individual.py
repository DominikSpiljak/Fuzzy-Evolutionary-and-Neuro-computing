import numpy as np
import pickle

class Individual:

    def __init__(self, value, neural_net):
        self.value = np.array(value)
        self.error = neural_net.calculate_error(self.value)

    def __eq__(self, other):
        return np.all(self.value == other.value)

    def __str__(self):
        return "Error = {}".format(self.error)

    def save_individual(self, filename):
        with open(filename, 'wb') as out:
            pickle.dump(self, out)