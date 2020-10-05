from domain import Domain

class SimpleDomain(Domain):
    first: int
    second: int

    def __init__(self, first, second):
        self.super(SimpleDomain, self)
        self.first = first
        self.second = second

    def getCardinality(self):
        return self.second - self.first

    def getComponent(self, index):
        pass

    def getNumberOfComponents(self):
        pass
