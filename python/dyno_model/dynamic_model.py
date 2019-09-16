from abc import ABC, abstractmethod

class DynamicModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def compute_jacobian(self):
        pass

    @abstractmethod
    def compute_cost(self):
        pass

    @abstractmethod
    def Ju(self):
        pass

    @abstractmethod
    def Fu(self):
        pass

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def compute_hessian(self):
        pass
