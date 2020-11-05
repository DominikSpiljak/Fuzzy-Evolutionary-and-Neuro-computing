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


class AccelFuzzySystem(FuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)

        self.velocity_domain = Domain.intRange(0, 1300)
        self.acceleration_domain = Domain.intRange(-35, 35)
        self.distance_domain = Domain.intRange(0, 1301)

        self.velocity = [
            # Sporo
            CalculatedFuzzySet(self.velocity_domain,
                               StandardFuzzySets.lFunction(40, 75)),

            # Normalno
            CalculatedFuzzySet(self.velocity_domain,
                               StandardFuzzySets.lambdaFunction(60, 75, 90)),

            # Brzo
            CalculatedFuzzySet(self.velocity_domain,
                               StandardFuzzySets.gammaFunction(75, 90))
        ]

        self.distance = [
            # Jako Blizu
            CalculatedFuzzySet(self.distance_domain,
                               StandardFuzzySets.lFunction(15, 30)),
            # Blizu
            CalculatedFuzzySet(self.distance_domain,
                               StandardFuzzySets.lFunction(30, 45))
        ]

        self.acceleration = [
            # Uspori
            CalculatedFuzzySet(self.acceleration_domain,
                               StandardFuzzySets.lFunction(-20, -10)),
            # Slabo uspori
            CalculatedFuzzySet(self.acceleration_domain,
                               StandardFuzzySets.lambdaFunction(-15, -10, 0)),
            # Slabo ubrzaj
            CalculatedFuzzySet(self.acceleration_domain,
                               StandardFuzzySets.lambdaFunction(0, 10, 15)),
            # Ubrzaj
            CalculatedFuzzySet(self.acceleration_domain,
                               StandardFuzzySets.gammaFunction(10, 20)),
        ]

        self.rules = [
            # [L, D, LK, DK, V, S]
            [[None, None, None, None, self.velocity[0], None], self.acceleration[3]],
            [[None, None, None, None, self.velocity[2], None], self.acceleration[0]],
            [[None, None, self.distance[1], None, None, None], self.acceleration[0]],
            [[None, None, None, self.distance[1], None, None], self.acceleration[0]],
            [[self.distance[0], None, None, None, None, None], self.acceleration[3]],
            [[None, self.distance[0], None, None, None, None], self.acceleration[3]],
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

    def evaluate_rule(self, singletons, rule, and_operator, or_operator):
        conclusion = self.conclude_rule(
            singletons, rule, and_operator, or_operator)
        return conclusion, self.defuzzier(conclusion)


class AccelFuzzySystemMin(AccelFuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)
        self.and_operator = min
        self.or_operator = max

    def conclude(self, L, D, LK, DK, V, S, return_conclusion=False):

        singletons = list(map(DomainElement.of, [L, D, LK, DK, V, S]))

        # Prazan fuzzy_set
        conclusion = MutableFuzzySet(self.acceleration_domain)

        for rule in self.rules:
            # Pripadnost akcedenta
            y_cutoff = self.conclude_rule(
                singletons, rule, self.and_operator, self.or_operator)

            conclusion = Operations.binaryOperation(
                conclusion, y_cutoff, Operations.zadehOr())

        if return_conclusion:
            return conclusion, int(self.defuzzier.decode(conclusion))
        else:
            return int(self.defuzzier.decode(conclusion))


class AccelFuzzySystemProduct(AccelFuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)
        self.and_operator = min
        self.or_operator = lambda l: reduce(lambda x, y: x * y, l)

    def conclude(self, L, D, LK, DK, V, S, return_conclusion=False):

        singletons = list(map(DomainElement.of, [L, D, LK, DK, V, S]))

        # Prazan fuzzy_set
        conclusion = MutableFuzzySet(self.acceleration_domain)

        for rule in self.rules:
            # Pripadnost akcedenta
            y_cutoff = self.conclude_rule(
                singletons, rule, self.and_operator, self.or_operator)

            conclusion = Operations.binaryOperation(
                conclusion, y_cutoff, Operations.zadehOr())

        if return_conclusion:
            return conclusion, int(self.defuzzier.decode(conclusion))
        else:
            return int(self.defuzzier.decode(conclusion))


if __name__ == "__main__":
    from defuzziers import COADefuzzifier
    defuz = COADefuzzifier()
    sys = AccelFuzzySystemMin(defuz)
    print(sys.conclude(L=20, D=50, LK=40, DK=60, V=0, S=1))
