from domain_element import DomainElement
from operations import Operations
from fuzzyset_mutable import MutableFuzzySet
from domain import Domain

class Relations:

    @staticmethod
    def isSymmetric(fuzzyset):
        for element in fuzzyset.getDomain():
            if fuzzyset.getValueAt(element) != fuzzyset.getValueAt(DomainElement.of(*element.values[::-1])):
                return False
        return True

    @staticmethod
    def isReflexive(fuzzyset):
        for element in fuzzyset.getDomain():
            if element.values[0] == element.values[1] and fuzzyset.getValueAt(element) != 1:
                return False
        return True 

    @staticmethod
    def isMaxMinTransitive(fuzzyset):

        domain_u = fuzzyset.getDomain().getComponent(0)
        for x in domain_u:
            x = x.values[0]
            for z in domain_u:
                z = z.values[0]
                connections = []
                for y in domain_u:
                    y = y.values[0]
                    connections.append(min(fuzzyset.getValueAt(DomainElement.of(x, y)), fuzzyset.getValueAt(DomainElement.of(y, z))))
                if max(connections) > fuzzyset.getValueAt(DomainElement.of(x, z)):
                    return False
        return True

    @staticmethod
    def compositionOfBinaryRelations(fuzzyset1, fuzzyset2):
        domain1_1 = fuzzyset1.getDomain().getComponent(0)
        domain2_2 = fuzzyset2.getDomain().getComponent(1)
        domain_u = Domain.combine(domain1_1, domain2_2)
        composition = MutableFuzzySet(domain_u)
        domain1_2 = fuzzyset1.getDomain().getComponent(1)

        for x in domain1_1:
            x = x.values[0]
            for z in domain2_2:
                z = z.values[0]
                connections = []
                for y in domain1_2:
                    y = y.values[0]
                    connections.append(min(fuzzyset1.getValueAt(DomainElement.of(x, y)), fuzzyset2.getValueAt(DomainElement.of(y, z))))
                composition.set(DomainElement.of(x, z), max(connections))
        return composition

    @staticmethod
    def isFuzzyEquivalence(fuzzyset):
        pass

    @staticmethod
    def isUtimesURelation(fuzzyset):
        domain1 = fuzzyset.getDomain().getComponent(0)
        domain2 = fuzzyset.getDomain().getComponent(1)
        if domain2.second == domain1.second and domain2.first == domain1.first:
            return True
        return False