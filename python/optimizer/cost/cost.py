from abc import ABC, abstractmethod

class Cost(ABC):
    def __init__(self, dimensions=2):
        self.cost = 0.0
        self.dimensions = 2
        super().__init__()

    @abstractmethod
    def compute_cost(self):
        pass
