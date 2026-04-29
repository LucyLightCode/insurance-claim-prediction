# api/main.py
"""
FastAPI application for insurance claim prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import BuildingInput, PredictionOutput, HealthCheck
from src.inference import ClaimPredictor

# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Insurance Claim Risk Prediction API",
    description="Predicts the probability of insurance claims for buildings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP EVENT
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    global predictor
    try:
        predictor = ClaimPredictor()
        print("\n" + "="*70)
        print("✓ API STARTED SUCCESSFULLY")
        print("="*70)
        print(f"  Model Version: {predictor.metadata['model_version']}")
        print(f"  Documentation: http://localhost:8000/docs")
        print("="*70 + "\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint.
    Returns API status and model information.
    """
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_version": predictor.metadata['model_version'] if predictor else "unknown"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(building: BuildingInput):
    """
    Predict insurance claim risk for a building.
    
    **Parameters:**
    - All building characteristics (see example in schema)
    
    **Returns:**
    - claim_probability: Probability of claim (0-1)
    - risk_category: Low / Medium / High / Very High
    - recommendation: Underwriting action
    - model_version: Model version used
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please restart the API."
        )
    
    try:
        # Convert Pydantic model to dict
        building_data = building.model_dump(by_alias=True)
        
        # Get prediction
        result = predictor.predict(building_data)
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict")
async def batch_predict(buildings: list[BuildingInput]):
    """
    Predict risk for multiple buildings at once.
    
    **Parameters:**
    - List of building data
    
    **Returns:**
    - List of predictions
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for building in buildings:
            building_data = building.model_dump(by_alias=True)
            result = predictor.predict(building_data)
            results.append(result)
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# RUN WITH: uvicorn api.main:app --reload
# ══════════════════════════════════════════════════════════════════════════════