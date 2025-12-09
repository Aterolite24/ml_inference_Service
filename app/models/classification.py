import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from app.models.registry import BaseModel, ModelRegistry

@ModelRegistry.register("logistic_regression")
class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = LogisticRegression(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("knn_classifier")
class KNNModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = KNeighborsClassifier(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("decision_tree_classifier")
class DecisionTreeModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = DecisionTreeClassifier(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("random_forest_classifier")
class RandomForestClfModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = RandomForestClassifier(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("gradient_boosting_classifier")
class GradientBoostingClfModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = GradientBoostingClassifier(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
