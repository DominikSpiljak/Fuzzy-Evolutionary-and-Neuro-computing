import abc


class Defuzzier:

    @abc.abstractclassmethod
    def decode(self, element, value):
        pass


class COADefuzzifier(Defuzzier):

    def decode(self, fuzzy_set):
        if sum([fuzzy_set.getValueAt(member) for member in fuzzy_set.getDomain()]) == 0:
            return 0
        return sum([member.values[0] * fuzzy_set.getValueAt(member) for member in fuzzy_set.getDomain()]) / sum([fuzzy_set.getValueAt(member) for member in fuzzy_set.getDomain()])
