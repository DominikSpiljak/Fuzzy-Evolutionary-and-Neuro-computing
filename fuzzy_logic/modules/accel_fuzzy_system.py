from fuzzy_system import FuzzySystem
from fuzzyset_calculated import CalculatedFuzzySet
from fuzzyset_mutable import MutableFuzzySet
from fuzzyset_standard import StandardFuzzySets
from domain import Domain
from domain_element import DomainElement
from operations import Operations
from int_unaryfunction import IntUnaryFunction
import abc


class AccelFuzzySystem(FuzzySystem):

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

        speed_domain = Domain.intRange(0, 1301)
        self.speed_values = [
            # Jako sporo
            CalculatedFuzzySet(
                speed_domain, StandardFuzzySets.lFunction(5, 10)),
            # Dosta sporo
            CalculatedFuzzySet(
                speed_domain, StandardFuzzySets.lambdaFunction(5, 15, 25)),
            # Normalno
            CalculatedFuzzySet(
                speed_domain, StandardFuzzySets.lambdaFunction(20, 30, 40)),
            # Dosta brzo
            CalculatedFuzzySet(
                speed_domain, StandardFuzzySets.lambdaFunction(30, 40, 50)),
            # Jako brzo
            CalculatedFuzzySet(
                speed_domain, StandardFuzzySets.gammaFunction(40, 60))
        ]

        accel_domain = Domain.intRange(-50, 50)
        self.accel_values = [
            # Jako uspori
            CalculatedFuzzySet(
                accel_domain, StandardFuzzySets.lFunction(-20, -10)),
            # Slabo uspori
            CalculatedFuzzySet(
                accel_domain, StandardFuzzySets.lambdaFunction(-15, -7, 0)),
            # Neutralno
            CalculatedFuzzySet(
                accel_domain, StandardFuzzySets.lambdaFunction(-5, 0, 5)),
            # Slabo ubrzaj
            CalculatedFuzzySet(
                accel_domain, StandardFuzzySets.lambdaFunction(0, 7, 15)),
            # Jako ubrzaj
            CalculatedFuzzySet(
                accel_domain, StandardFuzzySets.gammaFunction(10, 20))
        ]

    @ abc.abstractclassmethod
    def conclude(self, L, D, LK, DK, V, S):
        pass


class AccelFuzzySystemMin(AccelFuzzySystem):

    def __init__(self, defuzzier):
        super().__init__(defuzzier)
        self.rules = [
            # Ako smo jako blizu za LK i idemo jako brzo, jako uspori
            # Ako smo jako blizu za LK i idemo dosta brzo, jako uspori
            [self.LKDK_distances[0], self.speed_values[4], self.accel_values[0]],
            [self.LKDK_distances[0], self.speed_values[3], self.accel_values[0]],
            # Ako smo dosta blizu LK i idemo jako brzo, slabo uspori
            # Ako smo dosta blizu LK i idemo jako sporo, slabo ubrzaj
            # Ako smo dosta blizu LK i idemo dosta sporo, slabo ubrzaj
            [self.LKDK_distances[1], self.speed_values[4], self.accel_values[1]],
            [self.LKDK_distances[1], self.speed_values[0], self.accel_values[3]],
            # Ako smo relativno blizu Lk i idemo jako sporo, jako ubrzaj
            # Ako smo relativno blizu Lk i idemo dosta sporo, slabo ubrzaj
            [self.LKDK_distances[2], self.speed_values[0], self.accel_values[3]],
            [self.LKDK_distances[2], self.speed_values[1], self.accel_values[4]],
            # Ako smo daleko od Lk i idemo jako sporo, jako ubrzaj
            # Ako smo daleko od Lk i idemo dosta sporo, jako ubrzaj
            # Ako smo daleko od Lk i idemo dosta sporo, slabo ubrzaj
            [self.LKDK_distances[3], self.speed_values[0], self.accel_values[4]],
            [self.LKDK_distances[3], self.speed_values[1], self.accel_values[4]],

            # Isto za i DK
            # Ako smo jako blizu za DK i idemo jako brzo, jako uspori
            # Ako smo jako blizu za DK i idemo dosta brzo, jako uspori
            [self.LKDK_distances[0], self.speed_values[4], self.accel_values[0]],
            [self.LKDK_distances[0], self.speed_values[3], self.accel_values[0]],
            # Ako smo dosta blizu DK i idemo jako brzo, slabo uspori
            # Ako smo dosta blizu DK i idemo jako sporo, slabo ubrzaj
            [self.LKDK_distances[1], self.speed_values[4], self.accel_values[1]],
            [self.LKDK_distances[1], self.speed_values[0], self.accel_values[3]],
            # Ako smo relativno blizu DK i idemo jako sporo, jako ubrzaj
            # Ako smo relativno blizu DK i idemo dosta sporo, slabo ubrzaj
            [self.LKDK_distances[2], self.speed_values[0], self.accel_values[3]],
            [self.LKDK_distances[2], self.speed_values[1], self.accel_values[4]],
            # Ako smo daleko od DK i idemo jako sporo, jako ubrzaj
            # Ako smo daleko od DK i idemo dosta sporo, jako ubrzaj
            [self.LKDK_distances[3], self.speed_values[0], self.accel_values[4]],
            [self.LKDK_distances[3], self.speed_values[1], self.accel_values[4]],
        ]

    def conclude(self, L, D, LK, DK, V, S):
        LK_element = DomainElement.of(LK)
        DK_element = DomainElement.of(DK)
        V_element = DomainElement.of(V)
        L_element = DomainElement.of(L)
        D_element = DomainElement.of(D)

        # Prazan fuzzy_set
        conclusion = MutableFuzzySet(self.rules[0][2].getDomain())

        for i, rule in enumerate(self.rules):
            if i < 8:
                value = min([rule[0].getValueAt(LK_element),
                             rule[1].getValueAt(LK_element)])
            else:
                value = min([rule[0].getValueAt(DK_element),
                             rule[1].getValueAt(V_element)])

            # Pripadnost akcedenta
            cutoff = CalculatedFuzzySet(
                rule[2].getDomain(), IntUnaryFunction(lambda x: value))

            # Get y with cutoff
            y_cutoff = Operations.binaryOperation(
                cutoff, rule[2], Operations.zadehAnd())

            conclusion = Operations.binaryOperation(
                conclusion, y_cutoff, Operations.zadehOr())

        return int(self.defuzzier.decode(conclusion))
