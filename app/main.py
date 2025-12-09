from fastapi import FastAPI
from app.api import routes
# Import models to ensure they register themselves
from app.models import regression, classification, clustering

app = FastAPI(title="ML Inference Service", description="A dynamic ML model inference service")

app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "ML Service is running"}
