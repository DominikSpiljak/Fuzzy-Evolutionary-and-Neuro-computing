import numpy as np

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
