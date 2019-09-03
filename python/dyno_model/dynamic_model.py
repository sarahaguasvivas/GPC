from abc import ABC, abstractmethod

class DynamicModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self):
        pass


    @abstractmethod
    def compute_hessian(self):
        pass
