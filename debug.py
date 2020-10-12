from domain import Domain
from domain_element import DomainElement
from fuzzyset_mutable import MutableFuzzySet
from fuzzyset_calculated import CalculatedFuzzySet
from fuzzyset_standard import StandardFuzzySets
from relations import Relations
from operations import Operations
from copy import deepcopy


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
    d = Domain.intRange(1, 6)
    d2 = Domain.combine(d, d)

    r1 = (MutableFuzzySet(d2).set(DomainElement.of(1,1), 1).set(DomainElement.of(2,2), 1)
            .set(DomainElement.of(3,3), 1).set(DomainElement.of(4,4), 1).set(DomainElement.of(5,5), 1)
            .set(DomainElement.of(3,1), 0.5).set(DomainElement.of(1,3), 0.5))


    r2 = (MutableFuzzySet(d2).set(DomainElement.of(1,1), 1).set(DomainElement.of(2,2), 1)
            .set(DomainElement.of(3,3), 1).set(DomainElement.of(4,4), 1).set(DomainElement.of(5,5), 1)
            .set(DomainElement.of(3,1), 0.5).set(DomainElement.of(1,3), 0.1))

    r3 = (MutableFuzzySet(d2).set(DomainElement.of(1,1), 1).set(DomainElement.of(2,2), 1)
            .set(DomainElement.of(3,3), 0.3).set(DomainElement.of(4,4), 1).set(DomainElement.of(5,5), 1)
            .set(DomainElement.of(1,2), 0.6).set(DomainElement.of(2,1), 0.6).set(DomainElement.of(2,3), 0.7)
            .set(DomainElement.of(3,2), 0.7).set(DomainElement.of(3,1), 0.5).set(DomainElement.of(1,3), 0.5))

    r4 = (MutableFuzzySet(d2).set(DomainElement.of(1,1), 1).set(DomainElement.of(2,2), 1)
            .set(DomainElement.of(3,3), 1).set(DomainElement.of(4,4), 1).set(DomainElement.of(5,5), 1)
            .set(DomainElement.of(1,2), 0.4).set(DomainElement.of(2,1), 0.4).set(DomainElement.of(2,3), 0.5)
            .set(DomainElement.of(3,2), 0.5).set(DomainElement.of(1,3), 0.4).set(DomainElement.of(3,1), 0.4))

    print("r1 je definiran nad UxU? ", Relations.isUtimesURelation(r1))
    print("r1 je simetrična? ", Relations.isSymmetric(r1))
    print("r2 je simetrična? ", Relations.isSymmetric(r2))
    print("r1 je refleksivna? ", Relations.isReflexive(r1))
    print("r3 je refleksivna? ", Relations.isReflexive(r3))
    print("r3 je max-min tranzitivna? ", Relations.isMaxMinTransitive(r3))
    print("r4 je max-min tranzitivna? ", Relations.isMaxMinTransitive(r4))

    u1 = Domain.intRange(1, 5)
    u2 = Domain.intRange(1, 4)
    u3 = Domain.intRange(1, 5)

    r1 = (MutableFuzzySet(Domain.combine(u1, u2)).set(DomainElement.of(1,1), 0.3).set(DomainElement.of(1,2), 1)
                .set(DomainElement.of(3,3), 0.5).set(DomainElement.of(4,3), 0.5))
    r2 = (MutableFuzzySet(Domain.combine(u2, u3)).set(DomainElement.of(1,1), 1).set(DomainElement.of(2,1), 0.5)
                .set(DomainElement.of(2,2), 0.7).set(DomainElement.of(3,3), 1).set(DomainElement.of(3,4), 0.4))

    r1r2 = Relations.compositionOfBinaryRelations(r1, r2)

    for domainElement in r1r2.getDomain():
        print('mu({}) = {}'.format(domainElement, r1r2.getValueAt(domainElement)))

    u = Domain.intRange(1, 5)
    r = (MutableFuzzySet(Domain.combine(u, u)).set(DomainElement.of(1,1), 1).set(DomainElement.of(2,2), 1)
            .set(DomainElement.of(3,3), 1).set(DomainElement.of(4,4), 1)
            .set(DomainElement.of(1,2), 0.3).set(DomainElement.of(2,1), 0.3).set(DomainElement.of(2,3), 0.5)
            .set(DomainElement.of(3,2), 0.5).set(DomainElement.of(3,4), 0.2).set(DomainElement.of(4,3), 0.2))
    
    r2 = MutableFuzzySet(Domain.combine(u, u))
    r2.__dict__ = deepcopy(r.__dict__)

    print("Početna relacija je neizrazita relacija ekvivalencije? ", Relations.isFuzzyEquivalence(r2))
    print()

    for i in range(3):
        r2 = Relations.compositionOfBinaryRelations(r2, r)

        print("Broj odrađenih kompozicija:",  i, ". Relacija je:")

        for element in r2.getDomain():
            print("mu({}) = {}".format(element, r2.getValueAt(element)))

        print("Ova relacija je neizrazita relacija ekvivalencije? ", Relations.isFuzzyEquivalence(r2))
        print()