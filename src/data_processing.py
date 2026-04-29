# src/data_processing.py
"""
Data preprocessing and feature engineering functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw building data
    
    Returns:
    --------
    pd.DataFrame with engineered features
    """
    df = df.copy()
    
    # ── 1. Windows Unknown Flag ──────────────────────────────────────────
    df['NumberOfWindows'] = pd.to_numeric(df['NumberOfWindows'], errors='coerce')
    df['Windows_Unknown'] = df['NumberOfWindows'].isna().astype(int)
    df['NumberOfWindows'].fillna(df['NumberOfWindows'].median(), inplace=True)
    
    # ── 2. Building Age ──────────────────────────────────────────────────
    df['Building_Age'] = (
        df['YearOfObservation'] - df['Date_of_Occupancy']
    ).clip(lower=0)
    
    # ── 3. Log Building Dimension ────────────────────────────────────────
    df['Log_Building_Dimension'] = np.log1p(df['Building Dimension'])
    
    # ── 4. Structural Risk Flags ─────────────────────────────────────────
    df['Not_Painted'] = df['Building_Painted'].map({'V': 0, 'N': 1})
    df['Not_Fenced'] = df['Building_Fenced'].map({'V': 0, 'N': 1})
    df['No_Garden'] = df['Garden'].map({'V': 0, 'O': 1})
    df['Structural_Risk_Score'] = (
        df['Not_Painted'] + df['Not_Fenced'] + df['No_Garden']
    )
    
    # ── 5. Urban Flag ────────────────────────────────────────────────────
    df['Urban_Flag'] = df['Settlement'].map({'U': 1, 'R': 0})
    
    # ── 6. Age-Risk Interaction ──────────────────────────────────────────
    df['Age_Risk_Interaction'] = (
        df['Building_Age'] * df['Structural_Risk_Score']
    )
    
    return df


def apply_geo_encoding(
    df: pd.DataFrame,
    geo_map: Dict[str, float],
    global_mean: float
) -> pd.DataFrame:
    """
    Apply geographic target encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Geo_Code column
    geo_map : dict
        Mapping of Geo_Code to claim rate
    global_mean : float
        Fallback value for unseen geo codes
    
    Returns:
    --------
    pd.DataFrame with Geo_Code_TargetEnc column
    """
    df = df.copy()
    df['Geo_Code_TargetEnc'] = df['Geo_Code'].map(geo_map).fillna(global_mean)
    return df