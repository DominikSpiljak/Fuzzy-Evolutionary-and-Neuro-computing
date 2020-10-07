from fuzzyset_interface import FuzzySetInterface


class MutableFuzzySet(FuzzySetInterface):

    memberships: list

    def __init__(self, domain):
        super(MutableFuzzySet, self).__init__()
        self.memberships = [0.0 for element in range(domain.getCardinality())]
        self.domain = domain

    def getDomain(self):
        return self.domain

    def getValueAt(self, element):
        return self.memberships[self.domain.indexOfElement(element)]

    def set(self, element, value):
        self.memberships[self.domain.indexOfElement(element)] = value
        return self
