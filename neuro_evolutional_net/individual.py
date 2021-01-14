import numpy as np
import pickle
from copy import deepcopy


class Individual:

    def __init__(self, value, neural_net):
        self.value = np.array(value.copy())
        self.w_type_1, self.s_type_1, self.w, self.b = neural_net.decode_params(
            self.value)
        self.error = neural_net.calculate_error(
            [self.w_type_1, self.s_type_1, self.w, self.b])

    def get_value(self):
        return self.value.copy()

    def __eq__(self, other):
        return np.all(self.value == other.value)

    def __str__(self):
        return "Error = {}".format(self.error)

    def save_individual(self, filename):
        with open(filename, 'wb') as out:
            pickle.dump(self, out)
