from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class ModelConfig(BaseModel):
    model_type: str = Field(..., description="Type of model to use (e.g., 'linear_regression', 'kmeans')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters for the model")

class TrainRequest(BaseModel):
    model_id: str = Field(..., description="Unique identifier for this model instance")
    config: ModelConfig
    X: List[List[float]] = Field(..., description="Feature matrix")
    y: Optional[Union[List[float], List[int]]] = Field(None, description="Target vector (optional for clustering)")

class PredictRequest(BaseModel):
    model_id: str = Field(..., description="ID of the trained model to use")
    X: List[List[float]] = Field(..., description="Feature matrix for prediction")

class PredictionResponse(BaseModel):
    model_id: str
    predictions: List[Union[float, int]]
