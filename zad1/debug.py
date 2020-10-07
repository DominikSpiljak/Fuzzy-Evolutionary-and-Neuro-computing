from domain import Domain
from domain_element import DomainElement
from fuzzyset_mutable import MutableFuzzySet
from fuzzyset_calculated import CalculatedFuzzySet
from fuzzyset_standard import StandardFuzzySets
from operations import Operations


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
    set1 = MutableFuzzySet(d).set(DomainElement.of(0), 1.0).set(DomainElement.of(1), 0.8).set(
        DomainElement.of(2), 0.6).set(DomainElement.of(3), 0.4).set(DomainElement.of(4), 0.2)

    Debug.debug_print_fuzzyset(set1, "Set1:")
    notSet1 = Operations.unaryOperation(set1, Operations.zadehNot())
    Debug.debug_print_fuzzyset(notSet1, "notSet1")

    union = Operations.binaryOperation(
        set1, notSet1, Operations.zadehOr())
    Debug.debug_print_fuzzyset(union, "Set1 union notSet1:")

    hinters = Operations.binaryOperation(
        set1, notSet1, Operations.hamacherTNorm(1.0))

    Debug.debug_print_fuzzyset(
        hinters, "Set1 intersection with notSet1 using parameterised Hamacher T norm with parameter 1.0:")
