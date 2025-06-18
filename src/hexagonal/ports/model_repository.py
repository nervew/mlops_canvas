from abc import ABC, abstractmethod
from typing import Any


class ModelRepository(ABC):
    """Port for persisting and retrieving trained models."""

    @abstractmethod
    def save(self, model: Any) -> None:
        """Persist a trained model."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> Any:
        """Retrieve a trained model."""
        raise NotImplementedError
