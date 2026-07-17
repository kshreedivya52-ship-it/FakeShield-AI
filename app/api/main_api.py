import os
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# Import our modules
from app.explainability.shap_explainer import SHAPExplainer
from app.rag.rag_pipeline import RAGPipeline
from app.rag.retriever import NewsRetriever

app = FastAPI(
    title="FakeShield AI API",
    description="REST API for Explainable Fake News Detection and RAG-based Fact Checking",
    version="1.0.0",
)

# Request Models
class NewsRequest(BaseModel):
    text: str = Field(..., description="The news text to analyze")


class FeedbackRequest(BaseModel):
    text: str = Field(..., description="The analyzed news text")
    prediction: str = Field(..., description="The model's prediction (FAKE/REAL)")
    is_correct: bool = Field(..., description="Whether the user agrees with the prediction")
    comment: str = Field("", description="Optional comments from the user")


# Global instances of our processing engines (initialized lazily on startup)
explainer = None
rag_pipeline = None


@app.on_event("startup")
def startup_event():
    global explainer, rag_pipeline
    print("Initializing API services...")
    explainer = SHAPExplainer()
    retriever = NewsRetriever()
    rag_pipeline = RAGPipeline(retriever=retriever)
    print("API services successfully initialized.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "model_loaded": explainer is not None,
            "rag_loaded": rag_pipeline is not None,
        },
    }


@app.post("/predict")
def predict_news(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        # Explainer initialization is done internally
        res = explainer.explain(request.text)
        return {"prediction": res["prediction"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain")
def explain_news(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        res = explainer.explain(request.text)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")


@app.post("/rag")
def rag_verify(request: NewsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        res = rag_pipeline.generate_explanation(request.text)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution error: {str(e)}")


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    feedback_file = Path("outputs/feedback.json")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing feedback
    feedbacks = []
    if feedback_file.exists():
        try:
            with open(feedback_file, "r") as f:
                feedbacks = json.load(f)
        except Exception:
            feedbacks = []

    # Append new feedback
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "text": request.text,
        "prediction": request.prediction,
        "is_correct": request.is_correct,
        "comment": request.comment,
    }
    feedbacks.append(new_entry)

    try:
        with open(feedback_file, "w") as f:
            json.dump(feedbacks, f, indent=4)
        return {"status": "success", "message": "Feedback submitted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
