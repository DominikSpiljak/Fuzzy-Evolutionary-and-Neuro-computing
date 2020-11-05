from fuzzyset_mutable import MutableFuzzySet
from domain import Domain
from unaryfunction import UnaryFunction
from binaryfunction import BinaryFunction


class Operations:

    @staticmethod
    def unaryOperation(fuzzyset, unaryFunction):
        new_set = MutableFuzzySet(fuzzyset.getDomain())
        for element in fuzzyset.getDomain():
            new_set.set(element, unaryFunction.valueAt(
                fuzzyset.getValueAt(element)))
        return new_set

    @staticmethod
    def binaryOperation(fuzzySet1, fuzzySet2, binaryFunction):
        new_set = MutableFuzzySet(fuzzySet1.getDomain())
        for element in new_set.getDomain():
            new_set.set(element, binaryFunction.valueAt(
                fuzzySet1.getValueAt(element), fuzzySet2.getValueAt(element)))
        return new_set

    @staticmethod
    def zadehNot():
        return UnaryFunction(lambda x: round(1.0 - x, 16))

    @staticmethod
    def zadehAnd():
        return BinaryFunction(lambda x, y: min(x, y))

    @staticmethod
    def zadehOr():
        return BinaryFunction(lambda x, y: max(x, y))

    @staticmethod
    def hamacherTNorm(v):
        return BinaryFunction(lambda x, y: round((x * y) / (v + (1 - v) * (x + y - x * y)), 16))

    @staticmethod
    def hamacherSNorm(v):
        return BinaryFunction(lambda x, y: round((x + y - (2 - v) * x * y) / (1 - (1 - v) * x * y), 16))
