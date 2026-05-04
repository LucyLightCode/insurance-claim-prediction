# app_standalone.py
"""
Streamlit Web Interface for Insurance Claim Risk Prediction
STANDALONE VERSION - No external module imports
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Insurance Claim Risk Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    .risk-very-high {
        background-color: #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_artifacts():
    """Load model, geo map, and metadata."""
    try:
        model = joblib.load('models/insurance_claim_model_v1.pkl')
        geo_map = joblib.load('models/geo_encoding_map.pkl')
        with open('models/model_metadata.json') as f:
            metadata = json.load(f)
        return model, geo_map, metadata, None
    except Exception as e:
        return None, None, None, str(e)

model, geo_map, metadata, load_error = load_model_artifacts()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(building_data, geo_map, global_mean):
    """Apply feature engineering inline."""
    df = pd.DataFrame([building_data])
    
    # Windows
    df['NumberOfWindows'] = pd.to_numeric(df['NumberOfWindows'], errors='coerce')
    df['Windows_Unknown'] = df['NumberOfWindows'].isna().astype(int)
    df['NumberOfWindows'].fillna(3, inplace=True)
    
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
    df['Geo_Code_TargetEnc'] = df['Geo_Code'].map(geo_map).fillna(global_mean)
    
    return df

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🏠 Insurance Claim Risk Predictor</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    Predict the probability of insurance claims for buildings using machine learning.
    <br>
    <strong>Model Performance:</strong> ROC-AUC 0.85 | Accuracy 82%
</div>
""", unsafe_allow_html=True)

# Show model status
if model is None:
    st.error(f"❌ Failed to load model: {load_error}")
    st.info("Please ensure model files exist in the `models/` directory")
    st.stop()
else:
    st.success(f"✅ Model loaded successfully (v{metadata['model_version']})")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📋 Building Information")
    
    st.subheader("Basic Details")
    year_observation = st.number_input(
        "Year of Observation",
        min_value=2010, max_value=2030, value=2015, step=1
    )
    
    insured_period = st.number_input(
        "Insured Period (years)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.5
    )
    
    residential = st.selectbox(
        "Building Type",
        options=[0, 1],
        format_func=lambda x: "Non-Residential" if x == 0 else "Residential"
    )
    
    building_type = st.selectbox(
        "Building Category",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Type {x}"
    )
    
    st.subheader("Location")
    settlement = st.selectbox(
        "Settlement Type",
        options=["U", "R"],
        format_func=lambda x: "Urban" if x == "U" else "Rural"
    )
    
    geo_code = st.text_input("Geographic Code", value="1053")
    
    st.subheader("Physical Characteristics")
    building_dimension = st.number_input(
        "Building Dimension (m²)",
        min_value=100.0, max_value=5000.0, value=595.0, step=10.0
    )
    
    occupancy_year = st.number_input(
        "Year of Occupancy",
        min_value=1900, max_value=2030, value=1960, step=1
    )
    
    num_windows = st.text_input(
        "Number of Windows (or '.' if unknown)",
        value="."
    )
    
    st.subheader("Structural Condition")
    painted = st.selectbox(
        "Is the building painted?",
        options=["V", "N"],
        format_func=lambda x: "Yes" if x == "V" else "No"
    )
    
    fenced = st.selectbox(
        "Is the building fenced?",
        options=["V", "N"],
        format_func=lambda x: "Yes" if x == "V" else "No"
    )
    
    garden = st.selectbox(
        "Does it have a garden?",
        options=["V", "O"],
        format_func=lambda x: "Yes" if x == "V" else "No"
    )
    
    st.markdown("---")
    predict_button = st.button("🔮 Predict Risk", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA - PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

if predict_button:
    # Build input
    building_data = {
        'YearOfObservation': year_observation,
        'Insured_Period': insured_period,
        'Residential': residential,
        'Building_Painted': painted,
        'Building_Fenced': fenced,
        'Garden': garden,
        'Settlement': settlement,
        'Building Dimension': building_dimension,
        'Building_Type': building_type,
        'Date_of_Occupancy': occupancy_year,
        'NumberOfWindows': num_windows,
        'Geo_Code': geo_code
    }
    
    try:
        with st.spinner("Analyzing building risk..."):
            # Engineer features
            df = engineer_features(building_data, geo_map, metadata['global_mean_claim_rate'])
            
            # Select features
            X = df[metadata['numerical_features'] + metadata['categorical_features']]
            
            # Predict
            prob = model.predict_proba(X)[0, 1]
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Claim Probability", f"{prob:.2%}")
        
        if prob < 0.15:
            risk = "Low"
            rec = "Standard underwriting approval"
        elif prob < 0.30:
            risk = "Medium"
            rec = "Review building age and structural condition"
        elif prob < 0.50:
            risk = "High"
            rec = "Enhanced inspection required before approval"
        else:
            risk = "Very High"
            rec = "Deny or require premium surcharge of 20%+"
        
        col2.metric("Risk Category", risk)
        col3.metric("Model Version", metadata['model_version'])
        
        risk_class = risk.lower().replace(' ', '-')
        st.markdown(f"""
        <div class="risk-{risk_class}">
            <h3>📋 Recommendation</h3>
            <p style='font-size: 1.1rem; margin: 0;'>{rec}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 View Details"):
            st.json({
                'claim_probability': float(prob),
                'risk_category': risk,
                'recommendation': rec,
                'model_version': metadata['model_version']
            })
            
            st.subheader("Building Features")
            features_df = pd.DataFrame([
                {"Feature": "Building Age", "Value": f"{year_observation - occupancy_year} years"},
                {"Feature": "Settlement", "Value": "Urban" if settlement == "U" else "Rural"},
                {"Feature": "Dimension", "Value": f"{building_dimension} m²"},
                {"Feature": "Painted", "Value": "Yes" if painted == "V" else "No"},
                {"Feature": "Fenced", "Value": "Yes" if fenced == "V" else "No"},
                {"Feature": "Garden", "Value": "Yes" if garden == "V" else "No"},
            ])
            st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

else:
    st.info("👈 Enter building details in the sidebar and click **Predict Risk**")
    
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    This application uses a machine learning model trained on **7,000+ insurance records**.
    
    **Key Factors:**
    - 🏗️ Building age and structural condition
    - 📍 Geographic location and settlement type
    - 🏠 Building dimensions and type
    - 🛡️ Maintenance status (painted, fenced, garden)
    """)
    
    st.markdown("### 📈 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", "0.85", "+0.10")
    col2.metric("Accuracy", "82%", "+7%")
    col3.metric("Recall", "80%", "High")
    
    st.markdown("### 🔒 Risk Categories")
    st.markdown("""
    - **Low (<15%)**: Standard approval
    - **Medium (15-30%)**: Review condition
    - **High (30-50%)**: Enhanced inspection
    - **Very High (>50%)**: Deny or surcharge
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with ❤️ using Streamlit and Scikit-Learn | 
    <a href='https://insurance-claim-prediction-niha.onrender.com/docs' target='_blank'>API Docs</a>
</div>
""", unsafe_allow_html=True)