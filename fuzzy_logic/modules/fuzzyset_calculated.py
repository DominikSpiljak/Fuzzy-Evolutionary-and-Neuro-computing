from fuzzyset_interface import FuzzySetInterface


class CalculatedFuzzySet(FuzzySetInterface):

    def __init__(self, domain, function):
        self.domain = domain
        self.function = function

    def getDomain(self):
        return self.domain

    def getValueAt(self, element):
        return self.function.valueAt(element.values[0])
