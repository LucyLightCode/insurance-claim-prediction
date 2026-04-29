# src/config.py
"""
Configuration settings for Insurance Claim Prediction System.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

MODEL_PATH = MODEL_DIR / 'insurance_claim_model_v1.pkl'
GEO_MAP_PATH = MODEL_DIR / 'geo_encoding_map.pkl'
METADATA_PATH = MODEL_DIR / 'model_metadata.json'

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_VERSION = '1.0'
PREDICTION_THRESHOLD = 0.3

# ══════════════════════════════════════════════════════════════════════════════
# FEATURES
# ══════════════════════════════════════════════════════════════════════════════

NUMERICAL_FEATURES = [
    'Building_Age',
    'Log_Building_Dimension',
    'Insured_Period',
    'NumberOfWindows',
    'Windows_Unknown',
    'Structural_Risk_Score',
    'Not_Painted',
    'Not_Fenced',
    'No_Garden',
    'Age_Risk_Interaction',
    'YearOfObservation',
    'Geo_Code_TargetEnc',
    'Residential',
    'Urban_Flag'
]

CATEGORICAL_FEATURES = ['Building_Type']

# ══════════════════════════════════════════════════════════════════════════════
# RISK THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

RISK_THRESHOLDS = {
    'Low': 0.15,
    'Medium': 0.30,
    'High': 0.50
}

RISK_RECOMMENDATIONS = {
    'Low': 'Standard underwriting approval',
    'Medium': 'Review building age and structural condition',
    'High': 'Enhanced inspection required before approval',
    'Very High': 'Deny or require premium surcharge of 20%+'
}