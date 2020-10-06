from domain import Domain
from domain_element import DomainElement
from fuzzyset_mutuable import MutuableFuzzySet
from fuzzyset_calculated import CalculatedFuzzySet
from fuzzyset_standard import StandardFuzzySets


class Debug:

    @staticmethod
    def debug_print_domain(domain, heading_text=""):
        print(heading_text)
        for element in domain:
            print(element)
        print("Kardinalitet domene je: {}".format(domain.getCardinality()))
        print()

    @staticmethod
    def debug_print_fuzzyset(fuzzyset, heading_text=""):
        print(heading_text)
        for element in fuzzyset.getDomain():
            print("d({})={}".format(element, fuzzyset.getValueAt(element)))
        print()


if __name__ == "__main__":
    d = Domain.intRange(0, 11)
    set1 = MutuableFuzzySet(d).set(DomainElement.of(0), 1.0).set(DomainElement.of(1), 0.8).set(
        DomainElement.of(2), 0.6).set(DomainElement.of(3), 0.4).set(DomainElement.of(4), 0.2)
    Debug.debug_print_fuzzyset(set1, 'Set1')
    d2 = Domain.intRange(-5, 6)
    set2 = CalculatedFuzzySet(d2, StandardFuzzySets.lambdaFunction(
        d2.indexOfElement(DomainElement.of(-4)), d2.indexOfElement(DomainElement.of(0)), d2.indexOfElement(DomainElement.of(4))))
    Debug.debug_print_fuzzyset(set2, 'Set2')
