"""
main.py — FastAPI Model Serving
Serves the best_pipeline.pkl saved by discovery.py.

Usage:
    uvicorn main:app --reload

Then open: http://127.0.0.1:8000/docs  (Swagger UI)

Endpoints:
    GET  /         — health check
    GET  /info     — model metadata
    POST /predict  — single prediction
    POST /predict/batch — batch predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import os

# ============================================================
# App setup
# ============================================================

app = FastAPI(
    title="Wine Quality Classifier API",
    description=(
        "Predicts the quality class of a red wine sample "
        "(low / medium / high) using a Logistic Regression pipeline "
        "trained on the UCI Red Wine Quality dataset (1,599 samples)."
    ),
    version="1.0.0",
    contact={
        "name": "Data Engineering Assignment",
        "email": "student@example.com",
    },
)

# ============================================================
# Model loading
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "best_pipeline.pkl")
_artifact = None


def get_artifact():
    """Lazy-load the pipeline artifact."""
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. "
                "Run discovery.py first to generate it."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


# ============================================================
# Schemas
# ============================================================

class WineFeatures(BaseModel):
    """Physicochemical features of a single red wine sample."""
    fixed_acidity:        float = Field(..., ge=4.0,  le=16.0,  example=7.4,
                                        description="Fixed acidity (g/dm³)")
    volatile_acidity:     float = Field(..., ge=0.1,  le=1.6,   example=0.70,
                                        description="Volatile acidity (g/dm³)")
    citric_acid:          float = Field(..., ge=0.0,  le=1.0,   example=0.00,
                                        description="Citric acid (g/dm³)")
    residual_sugar:       float = Field(..., ge=1.0,  le=16.0,  example=1.9,
                                        description="Residual sugar (g/dm³)")
    chlorides:            float = Field(..., ge=0.01, le=0.62,  example=0.076,
                                        description="Chlorides (g/dm³)")
    free_sulfur_dioxide:  float = Field(..., ge=1.0,  le=75.0,  example=11.0,
                                        description="Free SO₂ (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=5.0,  le=290.0, example=34.0,
                                        description="Total SO₂ (mg/dm³)")
    density:              float = Field(..., ge=0.99, le=1.01,  example=0.9978,
                                        description="Density (g/cm³)")
    pH:                   float = Field(..., ge=2.7,  le=4.1,   example=3.51,
                                        description="pH")
    sulphates:            float = Field(..., ge=0.3,  le=2.1,   example=0.56,
                                        description="Sulphates (g/dm³)")
    alcohol:              float = Field(..., ge=8.0,  le=15.0,  example=9.4,
                                        description="Alcohol (% vol)")

    def to_array(self) -> np.ndarray:
        return np.array([[
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.residual_sugar,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.pH,
            self.sulphates,
            self.alcohol,
        ]])


class PredictionResponse(BaseModel):
    prediction:   str          = Field(..., example="medium",
                                       description="Predicted quality class")
    probabilities: dict        = Field(...,
                                       example={"high": 0.18, "low": 0.05, "medium": 0.77},
                                       description="Class probabilities")
    model:        str          = Field(..., example="Logistic Regression")
    features_used: int         = Field(..., example=11)


class BatchRequest(BaseModel):
    samples: List[WineFeatures]


class BatchResponse(BaseModel):
    predictions: List[str]
    count: int


# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["Health"])
def root():
    """Health-check endpoint."""
    return {"status": "ok", "service": "Wine Quality Classifier API", "version": "1.0.0"}


@app.get("/info", tags=["Model Info"])
def model_info():
    """Return metadata about the loaded model."""
    try:
        art = get_artifact()
        return {
            "model_type":    type(art["pipeline"].named_steps["model"]).__name__,
            "classes":       list(art["label_encoder"].classes_),
            "feature_names": art["feature_names"],
            "pipeline_steps": list(art["pipeline"].named_steps.keys()),
            "dataset":       "UCI Red Wine Quality (1,599 samples)",
            "cv_accuracy":   "0.6586 ± 0.0224  (5-fold stratified CV)",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(wine: WineFeatures):
    """
    Predict the quality class of a single red wine sample.

    Returns:
    - **prediction**: `low` | `medium` | `high`
    - **probabilities**: softmax scores per class
    """
    try:
        art = get_artifact()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    pipeline = art["pipeline"]
    le       = art["label_encoder"]

    import pandas as pd
    X = pd.DataFrame([wine.model_dump()], columns=art["feature_names"])
    pred_label = pipeline.predict(X)[0]

    # Probability estimates
    proba     = pipeline.predict_proba(X)[0]
    classes   = pipeline.classes_
    prob_dict = {cls: round(float(p), 4)
                 for cls, p in zip(classes, proba)}

    return PredictionResponse(
        prediction=pred_label,
        probabilities=prob_dict,
        model=type(pipeline.named_steps["model"]).__name__,
        features_used=X.shape[1],
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Predict quality classes for a batch of wine samples (up to 100).
    """
    if len(request.samples) > 100:
        raise HTTPException(status_code=400,
                            detail="Batch size must not exceed 100 samples.")
    try:
        art = get_artifact()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    pipeline = art["pipeline"]
    le       = art["label_encoder"]

    import pandas as pd
    X = pd.DataFrame([s.model_dump() for s in request.samples], columns=art["feature_names"])
    pred_labels = pipeline.predict(X).tolist()

    return BatchResponse(predictions=pred_labels, count=len(pred_labels))


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
