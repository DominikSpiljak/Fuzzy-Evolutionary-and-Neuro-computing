from domain_interface import DomainInterface
from domain_element import DomainElement
from debug import Debug
import itertools


class Domain(DomainInterface):

    def __init__(self, elements):
        self.elements = elements

    @staticmethod
    def intRange(start, end):
        return Domain([DomainElement([i]) for i in range(start, end)])
    
    @staticmethod
    def combine(first, other):
        domain_elements = []
        for element1, element2 in itertools.product(first.elements, other.elements):
            domain_elements.append(DomainElement(element1.values + element2.values))
        return Domain(domain_elements)
        
    def indexOfElement(self, domainElement):
        return self.elements.index(domainElement)
    
    def elementForIndex(self, index):
        return self.elements[index]

    def __iter__(self):
        return iter(self.elements)
        
if __name__ == "__main__":
    d1 = Domain.intRange(0, 5)
    d2 = Domain.intRange(6, 10)
    d3 = Domain.combine(d1, d2)
    d4 = Domain.intRange(11, 15)
    Debug.debug_print(Domain.combine(d3, d4), "d5 elements")
