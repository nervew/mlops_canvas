from abc import ABC, abstractmethod
class IValidator(ABC):
    @abstractmethod
    def validate(self, transformer, df, features_finales):
        pass
