import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from app.models.registry import BaseModel, ModelRegistry

@ModelRegistry.register("linear_regression")
class LinearRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression(**kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("random_forest_regressor")
class RandomForestRegModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Filter None values to let defaults take over
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = RandomForestRegressor(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("gradient_boosting_regressor")
class GradientBoostingRegModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = GradientBoostingRegressor(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@ModelRegistry.register("bagging_regressor")
class BaggingRegModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.model = BaggingRegressor(**valid_kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
