from abc import ABC, abstractmethod
import numpy as np

class Hessian(ABC):
    def __init__(self, dimensions=2):
        self.Hessian= np.empty((2, 2))
        super().__init__()

    @abstractmethod
    def update_hessian(self):
        pass
