from domain_interface import DomainInterface
from domain_element import DomainElement
from debug import Debug
import itertools


class Domain(DomainInterface):

    elements: list

    def __init__(self, elements):
        self.elements = elements

    @staticmethod
    def intRange(start, end):
        return SimpleDomain(start, end)

    @staticmethod
    def combine(first, other):
        return CompositeDomain([first, other])

    def indexOfElement(self, domainElement):
        return self.elements.index(domainElement)

    def elementForIndex(self, index):
        return self.elements[index]

    def __iter__(self):
        return iter(self.elements)


class SimpleDomain(Domain):
    first: int
    second: int

    def __init__(self, first, second):
        self.first = first
        self.second = second
        super(SimpleDomain, self).__init__(
            [DomainElement([i]) for i in range(first, second)])

    def getCardinality(self):
        return self.second - self.first

    def getComponent(self, index):
        return self

    def getNumberOfComponents(self):
        return 1


class CompositeDomain(Domain):
    components: list

    def __init__(self, components):
        domain_elements = []
        for component in itertools.product(*components):
            flattened_list = []
            for element in component:
                flattened_list.extend(element.values)
            domain_elements.append(DomainElement(flattened_list))
        self.components = components
        super(CompositeDomain, self).__init__(domain_elements)

    def getCardinality(self):
        return sum([component.getCardinality() for component in self.components])

    def getComponent(self, index):
        found = None
        for component in self.components:
            if index == 0:
                found = component.getComponent(index)
            elif component.getNumberOfComponents() <= index:
                index -= component.getNumberOfComponents()
                continue
            else:
                found = component.getComponent(index)
            if found:
                return found
        raise IndexError("Index out of range")

    def getNumberOfComponents(self):
        return sum([component.getNumberOfComponents() for component in self.components])


if __name__ == "__main__":
    d1 = Domain.intRange(0, 5)
    Debug.debug_print(d1, 'Elementi domene d1')
    d2 = Domain.intRange(0, 3)
    Debug.debug_print(d2, 'Elementi domene d2')
    d3 = Domain.combine(d1, d2)
    Debug.debug_print(d3, 'Elementi domene d3')
    d4 = Domain.combine(d3, d1)
    Debug.debug_print(d4, 'Elementi domene d4')

    Debug.debug_print(d4.getComponent(0))

    print()
