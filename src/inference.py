# src/inference.py
"""
Model inference and prediction functions.
"""

import joblib
import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path

# Import from our config and data_processing modules
from src.config import (
    MODEL_PATH, GEO_MAP_PATH, METADATA_PATH,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    RISK_THRESHOLDS, RISK_RECOMMENDATIONS
)
from src.data_processing import engineer_features, apply_geo_encoding


class ClaimPredictor:
    """
    Insurance claim risk predictor.
    
    Usage:
    ------
    predictor = ClaimPredictor()
    result = predictor.predict(building_data)
    """
    
    def __init__(self):
        """Load model and artifacts on initialization."""
        self.model = joblib.load(MODEL_PATH)
        self.geo_map = joblib.load(GEO_MAP_PATH)
        
        with open(METADATA_PATH) as f:
            self.metadata = json.load(f)
        
        self.global_mean = self.metadata['global_mean_claim_rate']
        print(f"✓ Model loaded: v{self.metadata['model_version']}")
    
    def predict(self, building_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict claim probability for a building.
        
        Parameters:
        -----------
        building_data : dict
            Building characteristics with keys:
            - YearOfObservation, Insured_Period, Residential,
              Building_Painted, Building_Fenced, Garden, Settlement,
              Building Dimension, Building_Type, Date_of_Occupancy,
              NumberOfWindows, Geo_Code
        
        Returns:
        --------
        dict
            {
                'claim_probability': float (0-1),
                'risk_category': str,
                'recommendation': str,
                'model_version': str
            }
        """
        # Convert to DataFrame
        df = pd.DataFrame([building_data])
        
        # Apply feature engineering
        df = engineer_features(df)
        df = apply_geo_encoding(df, self.geo_map, self.global_mean)
        
        # Select features in correct order
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        
        # Predict probability
        probability = self.model.predict_proba(X)[0, 1]
        
        # Categorize risk
        risk_category = self._categorize_risk(probability)
        recommendation = RISK_RECOMMENDATIONS[risk_category]
        
        return {
            'claim_probability': round(float(probability), 4),
            'risk_category': risk_category,
            'recommendation': recommendation,
            'model_version': self.metadata['model_version']
        }
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize probability into risk levels."""
        if probability < RISK_THRESHOLDS['Low']:
            return 'Low'
        elif probability < RISK_THRESHOLDS['Medium']:
            return 'Medium'
        elif probability < RISK_THRESHOLDS['High']:
            return 'High'
        else:
            return 'Very High'


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST (when running this file directly)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Initialize predictor
    predictor = ClaimPredictor()
    
    # Test building
    test_building = {
        'YearOfObservation': 2015,
        'Insured_Period': 1.0,
        'Residential': 0,
        'Building_Painted': 'N',
        'Building_Fenced': 'V',
        'Garden': 'V',
        'Settlement': 'U',
        'Building Dimension': 595.0,
        'Building_Type': 1,
        'Date_of_Occupancy': 1960.0,
        'NumberOfWindows': '.',
        'Geo_Code': '1053'
    }
    
    # Get prediction
    result = predictor.predict(test_building)
    
    # Display result
    print('\n' + '='*70)
    print('PREDICTION TEST')
    print('='*70)
    for key, value in result.items():
        print(f'  {key}: {value}')
    print('='*70)