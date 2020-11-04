class UnaryFunction:

    def __init__(self, function):
        self.function = function

    def valueAt(self, x):
        return self.function(x)
