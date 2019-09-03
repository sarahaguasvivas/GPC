from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def optimize(self, u_n):
        pass

