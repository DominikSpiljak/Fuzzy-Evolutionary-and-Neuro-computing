class BinaryFunction:

    def __init__(self, function):
        self.function = function

    def valueAt(self, x1, x2):
        return self.function(x1, x2)
