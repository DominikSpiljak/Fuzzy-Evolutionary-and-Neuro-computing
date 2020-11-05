from fuzzy_system import FuzzySystem
from fuzzyset_calculated import CalculatedFuzzySet
from fuzzyset_mutable import MutableFuzzySet
from fuzzyset_standard import StandardFuzzySets
from domain import Domain
from domain_element import DomainElement
from operations import Operations
from int_unaryfunction import IntUnaryFunction
from functools import reduce
import abc


class SteerFuzzySystem(FuzzySystem):

    def __init__(self, defuzzier):

        super().__init__(defuzzier)
        self.distance_domain = Domain.intRange(0, 1301)
        self.orientation_domain = Domain.intRange(0, 2)
        self.steer_domain = Domain.intRange(-90, 91)

        self.rotation = CalculatedFuzzySet(self.distance_domain,
                                           StandardFuzzySets.lFunction(0, 1))

        self.distance = [
            # Jako Blizu
            CalculatedFuzzySet(self.distance_domain,
                               StandardFuzzySets.lFunction(25, 35)),
            # Blizu
            CalculatedFuzzySet(self.distance_domain,
                               StandardFuzzySets.lFunction(40, 60))
        ]

        self.steer = [

            # Dosta Desno
            CalculatedFuzzySet(self.steer_domain,
                               StandardFuzzySets.lFunction(-90, -75)),

            # Dosta Lijevo
            CalculatedFuzzySet(self.steer_domain,
                               StandardFuzzySets.gammaFunction(75, 90)),

            # Jako desno
            CalculatedFuzzySet(self.steer_domain,
                               StandardFuzzySets.gammaFunction(89, 90)),
        ]

        self.rules = [
            # [L, D, LK, DK, V, S]
            [[None, None, self.distance[1], None, None, None],
             self.steer[0]],
            [[None, None, None, self.distance[1], None, None],
             self.steer[1]],
            [[self.distance[0], None, None, None, None, None],
             self.steer[0]],
            [[None, self.distance[0], None, None, None, None],
             self.steer[1]],
            [[None, None, None, None, None, self.rotation],
             self.steer[2]],
        ]

    def conclude_rule(self, singletons, rule, and_operator, or_operator):

        premises = rule[0]
        y = rule[1]

        premises_values = []
        for singleton, premise in zip(singletons, premises):
            if premise is None:
                premises_values.append(1)
            elif type(premise) is list:
                premises_values.append(or_operator(
                    [p.getValueAt(singleton) for p in premise]))
            else:
                premises_values.append(premise.getValueAt(singleton))

        value = and_operator(premises_values)

        # Pripadnost akcedenta
        cutoff = CalculatedFuzzySet(
            y.getDomain(), IntUnaryFunction(lambda x: value))

        return Operations.binaryOperation(
            cutoff, y, Operations.zadehAnd())

    @ abc.abstractclassmethod
    def conclude(self, L, D, LK, DK, V, S):
        pass


class SteerFuzzySystemMin(SteerFuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)
        self.and_operator = min
        self.or_operator = max

    def conclude(self, L, D, LK, DK, V, S):

        singletons = list(map(DomainElement.of, [L, D, LK, DK, V, S]))

        # Prazan fuzzy_set
        conclusion = MutableFuzzySet(self.steer_domain)

        for rule in self.rules:
            # Pripadnost akcedenta
            y_cutoff = self.conclude_rule(
                singletons, rule, self.and_operator, self.or_operator)

            conclusion = Operations.binaryOperation(
                conclusion, y_cutoff, Operations.zadehOr())

        return int(self.defuzzier.decode(conclusion))


class SteerFuzzySystemProduct(SteerFuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)
        self.and_operator = min
        self.or_operator = lambda l: reduce(lambda x, y: x * y, l)

    def conclude(self, L, D, LK, DK, V, S):

        singletons = list(map(DomainElement.of, [L, D, LK, DK, V, S]))

        # Prazan fuzzy_set
        conclusion = MutableFuzzySet(self.steer_domain)

        for rule in self.rules:
            # Pripadnost akcedenta
            y_cutoff = self.conclude_rule(
                singletons, rule, self.and_operator, self.or_operator)

            conclusion = Operations.binaryOperation(
                conclusion, y_cutoff, Operations.zadehOr())

        return int(self.defuzzier.decode(conclusion))


if __name__ == "__main__":
    from defuzziers import COADefuzzifier
    defuz = COADefuzzifier()
    sys = SteerFuzzySystemMin(defuz)
    print(sys.conclude(L=20, D=50, LK=30, DK=60, V=0, S=1))
