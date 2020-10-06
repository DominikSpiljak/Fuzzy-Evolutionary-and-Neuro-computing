from int_unaryfunction import IntUnaryFunction


class StandardFuzzySets:

    @staticmethod
    def lFunction(x1, x2):

        def function(x): return 1 if x < x1 else (
            (x2 - x)/(x2 - x1) if x1 <= x and x < x2 else 0)

        return IntUnaryFunction(function)

    @staticmethod
    def gammaFunction(x1, x2):

        def function(x): return 0 if x < x1 else (
            (x - x1)/(x2 - x1) if x1 <= x and x < x2 else 1)

        return IntUnaryFunction(function)

    @staticmethod
    def lambdaFunction(x1, x2, x3):

        def function(x): return 0 if x < x1 else (
            (x - x1)/(x2 - x1) if x1 <= x and x < x2 else (
                (x3 - x)/(x3 - x2) if x2 <= x and x < x3 else 0))

        return IntUnaryFunction(function)
