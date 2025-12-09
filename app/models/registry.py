from abc import ABC, abstractmethod
import numpy as np
import pickle
from typing import Any, Dict, Optional

class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs

    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name: str) -> Optional[Any]:
        return cls._registry.get(name)

    @classmethod
    def list_models(cls):
        return list(cls._registry.keys())
