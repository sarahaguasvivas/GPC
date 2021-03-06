from abc import ABC, abstractmethod
import numpy as np

class Cost(ABC):
    def __init__(self):
        self.cost = 0.0
        super().__init__()

    @abstractmethod
    def compute_cost(self):
        pass
