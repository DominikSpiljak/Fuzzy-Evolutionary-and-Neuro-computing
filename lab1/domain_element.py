class DomainElement:
    def __init__(self, values):
        self.values = values

    def getNumberOfComponents(self):
        return len(self.values)

    def getComponentValue(self, index):
        return self.values(index)

    @staticmethod
    def of(element):
        return DomainElement([element])

    def __eq__(self, other):
        return self.values == other.values

    def __str__(self):
        return str(self.values)
