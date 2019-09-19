from abc import ABC, abstractmethod

class Constraints(ABC):
    def __init__(self, s = 1e-10, r = 1, b = 1):
        # s-> sharpness of the corners of the constraint function
        # r-> range of the constraint
        # b-> offset to the range
        self.s = s
        self.r = r
        self.b = b
        super().__init__()
