from abc import ABC, abstractmethod

class RegressorInterface(ABC):

    @abstractmethod
    def fit_regression(self):
        pass