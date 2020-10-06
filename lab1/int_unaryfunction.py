class IntUnaryFunction:

    def __init__(self, function):
        self.function = function

    def valueAt(self, index):
        return self.function(index)
