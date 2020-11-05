import abc
from domain_element import DomainElement
from operations import Operations
from domain import Domain
from fuzzyset_mutable import MutableFuzzySet


class FuzzySystem:
    def __init__(self, defuzzier):
        self.defuzzier = defuzzier

    @abc.abstractclassmethod
    def conclude(self, L, D, LK, DK, V, S):
        pass
