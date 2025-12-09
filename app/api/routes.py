from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.requests import TrainRequest, PredictRequest, PredictionResponse
from app.models.registry import ModelRegistry
import numpy as np

router = APIRouter()

# In-memory store for trained models
# In a real app, this should be a persistent store or cache (Redis/DB)
trained_models = {}

@router.get("/models")
def list_supported_models():
    """List all supported model types."""
    return {"models": ModelRegistry.list_models()}

@router.post("/train")
def train_model(request: TrainRequest):
    """
    Train a model with the given configuration and data.
    Stores the trained model in memory under `model_id`.
    """
    model_cls = ModelRegistry.get_model(request.config.model_type)
    if not model_cls:
        raise HTTPException(status_code=400, detail=f"Model type '{request.config.model_type}' not found.")

    try:
        # Instantiate model
        model = model_cls(**request.config.params)
        
        # Convert data
        X = np.array(request.X)
        y = np.array(request.y) if request.y is not None else None
        
        # Train
        model.train(X, y)
        
        # Save to store
        trained_models[request.model_id] = model
        
        return {"message": f"Model '{request.model_id}' trained and stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    """
    Generate predictions using a trained model.
    """
    if request.model_id not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found. Please train it first.")
    
    try:
        model = trained_models[request.model_id]
        X = np.array(request.X)
        
        predictions = model.predict(X)
        
        return PredictionResponse(
            model_id=request.model_id,
            predictions=predictions.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
