
import joblib
import json
import pandas as pd
import numpy as np

def predict_claim_risk(building_data):
    """Predict insurance claim probability."""
    # Load artifacts
    model = joblib.load('models/insurance_claim_model_v1.pkl')
    geo_map = joblib.load('models/geo_encoding_map.pkl')
    with open('models/model_metadata.json') as f:
        metadata = json.load(f)
    
    # Feature engineering
    df = pd.DataFrame([building_data])
    
    # Windows
    df['NumberOfWindows'] = pd.to_numeric(df['NumberOfWindows'], errors='coerce')
    df['Windows_Unknown'] = df['NumberOfWindows'].isna().astype(int)
    df['NumberOfWindows'].fillna(df['NumberOfWindows'].median(), inplace=True)
    
    # Building age
    df['Building_Age'] = (df['YearOfObservation'] - df['Date_of_Occupancy']).clip(lower=0)
    
    # Log dimension
    df['Log_Building_Dimension'] = np.log1p(df['Building Dimension'])
    
    # Structural flags
    df['Not_Painted'] = df['Building_Painted'].map({'V': 0, 'N': 1})
    df['Not_Fenced'] = df['Building_Fenced'].map({'V': 0, 'N': 1})
    df['No_Garden'] = df['Garden'].map({'V': 0, 'O': 1})
    df['Structural_Risk_Score'] = df['Not_Painted'] + df['Not_Fenced'] + df['No_Garden']
    
    # Urban flag
    df['Urban_Flag'] = df['Settlement'].map({'U': 1, 'R': 0})
    
    # Interaction
    df['Age_Risk_Interaction'] = df['Building_Age'] * df['Structural_Risk_Score']
    
    # Geo encoding
    df['Geo_Code_TargetEnc'] = df['Geo_Code'].map(geo_map).fillna(metadata['global_mean_claim_rate'])
    
    # Select features
    X = df[metadata['numerical_features'] + metadata['categorical_features']]
    
    # Predict
    probability = model.predict_proba(X)[0, 1]
    
    # Categorize
    if probability < 0.15:
        risk = 'Low'
        rec = 'Standard approval'
    elif probability < 0.30:
        risk = 'Medium'
        rec = 'Review building condition'
    elif probability < 0.50:
        risk = 'High'
        rec = 'Enhanced inspection required'
    else:
        risk = 'Very High'
        rec = 'Deny or premium surcharge'
    
    return {
        'claim_probability': round(float(probability), 4),
        'risk_category': risk,
        'recommendation': rec
    }

if __name__ == '__main__':
    # Test
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
    
    result = predict_claim_risk(test_building)
    print(json.dumps(result, indent=2))
