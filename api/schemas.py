# api/schemas.py
"""
Request and response schemas for the API.
"""

from pydantic import BaseModel, Field


class BuildingInput(BaseModel):
    """Input schema for building data."""
    
    YearOfObservation: int = Field(..., ge=2010, le=2030)
    Insured_Period: float = Field(..., ge=0)
    Residential: int = Field(..., ge=0, le=1)
    Building_Painted: str = Field(..., pattern="^[VN]$")
    Building_Fenced: str = Field(..., pattern="^[VN]$")
    Garden: str = Field(..., pattern="^[VO]$")
    Settlement: str = Field(..., pattern="^[UR]$")
    Building_Dimension: float = Field(..., gt=0, alias="Building Dimension")
    Building_Type: int = Field(..., ge=1, le=4)
    Date_of_Occupancy: float = Field(..., ge=1800, le=2030)
    NumberOfWindows: str
    Geo_Code: str
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "YearOfObservation": 2015,
                "Insured_Period": 1.0,
                "Residential": 0,
                "Building_Painted": "N",
                "Building_Fenced": "V",
                "Garden": "V",
                "Settlement": "U",
                "Building Dimension": 595.0,
                "Building_Type": 1,
                "Date_of_Occupancy": 1960.0,
                "NumberOfWindows": ".",
                "Geo_Code": "1053"
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for predictions."""
    
    claim_probability: float
    risk_category: str
    recommendation: str
    model_version: str


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_version: str