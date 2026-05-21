from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .schemas import QARequest, QAResponse, HealthResponse
from .logger import log_prediction
import sys
sys.path.append(".")
from src.model.predict import GujaratiQAModel

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = GujaratiQAModel()
    yield
    model = None

app = FastAPI(
    title="Gujarati QA API",
    description="Question Answering in Gujarati using IndicBERT",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=QAResponse)
def predict(request: QARequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = model.answer(request.question, request.context)

    # Log every prediction for monitoring
    log_prediction(
        question=request.question,
        context=request.context,
        answer=result["answer"],
        confidence=result["confidence"]
    )

    return QAResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        question=request.question
    )