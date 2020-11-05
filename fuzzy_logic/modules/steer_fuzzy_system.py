from fuzzy_system import FuzzySystem
from fuzzyset_calculated import CalculatedFuzzySet
from fuzzyset_mutable import MutableFuzzySet
from fuzzyset_standard import StandardFuzzySets
from domain import Domain
from domain_element import DomainElement
from operations import Operations
from int_unaryfunction import IntUnaryFunction
import matplotlib.pyplot as plt
import abc


class SteerFuzzySystem(FuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)
        domain = Domain.intRange(0, 1301)
        self.LKDK_distances = [
            # Jako blizu
            CalculatedFuzzySet(
                domain, StandardFuzzySets.lFunction(25, 35)),
            # Dosta blizu
            CalculatedFuzzySet(
                domain, StandardFuzzySets.lambdaFunction(30, 40, 50)),
            # Relativno blizu
            CalculatedFuzzySet(
                domain, StandardFuzzySets.lambdaFunction(40, 60, 80)),
            # Daleko
            CalculatedFuzzySet(
                domain, StandardFuzzySets.gammaFunction(70, 120))
        ]

        self.LD_distances = [
            # Jako blizu
            CalculatedFuzzySet(
                domain, StandardFuzzySets.lFunction(20, 25)),
            # Dosta blizu
            CalculatedFuzzySet(
                domain, StandardFuzzySets.lambdaFunction(20, 25, 30)),
            # Relativno blizu
            CalculatedFuzzySet(
                domain, StandardFuzzySets.lambdaFunction(25, 40, 55)),
            # Daleko
            CalculatedFuzzySet(
                domain, StandardFuzzySets.gammaFunction(50, 120))
        ]

        steer_domain = Domain.intRange(-90, 91)
        self.steer_rotations = [
            # Jako desno
            CalculatedFuzzySet(
                steer_domain, StandardFuzzySets.lFunction(-80, -65)),
            # Dosta desno
            CalculatedFuzzySet(
                steer_domain, StandardFuzzySets.lambdaFunction(-70, -50, -30)),
            # Slabo desno
            CalculatedFuzzySet(
                steer_domain, StandardFuzzySets.lambdaFunction(-50, -30, -10)),
            # Neutralno
            CalculatedFuzzySet(
                steer_domain, StandardFuzzySets.lambdaFunction(-15, 0, 15)),
            # Slabo lijevo
            CalculatedFuzzySet(
                steer_domain, StandardFuzzySets.lambdaFunction(10, 30, 50)),
            # Dosta lijevo
            CalculatedFuzzySet(
                steer_domain, StandardFuzzySets.lambdaFunction(30, 50, 70)),
            # Jako lijevo
            CalculatedFuzzySet(
                steer_domain, StandardFuzzySets.gammaFunction(65, 80))
        ]

    @ abc.abstractclassmethod
    def conclude(self, L, D, LK, DK, V, S):
        pass


class SteerFuzzySystemMin(SteerFuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)
        self.rules = [
            # Ako smo jako blizu za LK, skreni jako desno
            # Ako smo dosta blizu za LK, skreni dosta desno
            # Ako smo relativno blizu za LK, skreni slabo desno
            [self.LKDK_distances[0], self.steer_rotations[0]],
            [self.LKDK_distances[1], self.steer_rotations[1]],
            [self.LKDK_distances[2], self.steer_rotations[2]],

            # Isto za i DK, samo skrecemo lijevo
            [self.LKDK_distances[0], self.steer_rotations[6]],
            [self.LKDK_distances[1], self.steer_rotations[5]],
            [self.LKDK_distances[2], self.steer_rotations[4]],

            # Ako smo jako blizu za L, skreni dosta desno
            [self.LD_distances[0], self.steer_rotations[0]],

            # Ako smo jako blizu za D, skreni dosta lijevo
            [self.LD_distances[0], self.steer_rotations[6]],
        ]

    def conclude(self, L, D, LK, DK, V, S):
        LK_element = DomainElement.of(LK)
        DK_element = DomainElement.of(DK)
        L_element = DomainElement.of(L)
        D_element = DomainElement.of(D)

        # Prazan fuzzy_set
        conclusion = MutableFuzzySet(self.rules[0][1].getDomain())

        for i, rule in enumerate(self.rules):
            if i < 3:
                value = rule[0].getValueAt(LK_element)
            elif i < 6:
                value = rule[0].getValueAt(DK_element)
            elif i < 7:
                value = rule[0].getValueAt(L_element)
            else:
                value = rule[0].getValueAt(D_element)

            # Pripadnost akcedenta
            cutoff = CalculatedFuzzySet(
                rule[1].getDomain(), IntUnaryFunction(lambda x: value))

            # Get y with cutoff
            y_cutoff = Operations.binaryOperation(
                cutoff, rule[1], Operations.zadehAnd())

            conclusion = Operations.binaryOperation(
                conclusion, y_cutoff, Operations.zadehOr())

        return int(self.defuzzier.decode(conclusion))


class SteerFuzzySystemProduct(SteerFuzzySystem):
    def __init__(self, defuzzier):
        super().__init__(defuzzier)

    def conclude(self, L, D, LK, DK, V, S):
        pass


if __name__ == "__main__":
    from defuzziers import COADefuzzifier
    defuz = COADefuzzifier()
    sys = SteerFuzzySystemMin(defuz)
    print(sys.conclude(L=20, D=50, LK=40, DK=60, V=0, S=1))
