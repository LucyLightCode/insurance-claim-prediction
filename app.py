# app.py
"""
Streamlit Web Interface for Insurance Claim Risk Prediction
"""

import streamlit as st
import pandas as pd
import json
from src.inference import ClaimPredictor

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
def load_predictor():
    """Load model once and cache it."""
    return ClaimPredictor()

try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    model_loaded = False

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
# MAIN AREA - PREDICTION RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if predict_button and model_loaded:
    # Build input dictionary
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
    
    # Get prediction
    with st.spinner("Analyzing building risk..."):
        result = predictor.predict(building_data)
    
    # Display results
    st.success("✅ Prediction Complete!")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Claim Probability",
            value=f"{result['claim_probability']:.2%}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Risk Category",
            value=result['risk_category'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="Model Version",
            value=result['model_version'],
            delta=None
        )
    
    # Risk-based styling
    risk_category = result['risk_category'].lower().replace(' ', '-')
    
    st.markdown(f"""
    <div class="risk-{risk_category}">
        <h3>📋 Recommendation</h3>
        <p style='font-size: 1.1rem; margin: 0; color: green;'>{result['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show detailed breakdown
    with st.expander("📊 View Detailed Analysis"):
        st.json(result)
        
        # Show building features
        st.subheader("Building Features Used in Prediction")
        
        features_df = pd.DataFrame([
            {"Feature": "Building Age", "Value": f"{year_observation - occupancy_year} years"},
            {"Feature": "Settlement Type", "Value": "Urban" if settlement == "U" else "Rural"},
            {"Feature": "Building Dimension", "Value": f"{building_dimension} m²"},
            {"Feature": "Painted", "Value": "Yes" if painted == "V" else "No"},
            {"Feature": "Fenced", "Value": "Yes" if fenced == "V" else "No"},
            {"Feature": "Has Garden", "Value": "Yes" if garden == "V" else "No"},
        ])
        
        st.dataframe(features_df, use_container_width=True, hide_index=True)

elif predict_button and not model_loaded:
    st.error("❌ Model not loaded. Please check the console for errors.")

else:
    # Show example/instructions when no prediction yet
    st.info("👈 Enter building details in the sidebar and click **Predict Risk** to get started.")
    
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    This application uses a machine learning model trained on **7,000+ insurance records** to predict 
    the probability of a claim being filed for a building.
    
    **Key Factors Considered:**
    - 🏗️ Building age and structural condition
    - 📍 Geographic location and settlement type
    - 🏠 Building dimensions and type
    - 🛡️ Maintenance status (painted, fenced, garden)
    """)
    
    st.markdown("### 📈 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC Score", "0.85", "+0.10 vs baseline")
    col2.metric("Accuracy", "82%", "+7% vs baseline")
    col3.metric("Recall", "80%", "High claim detection")
    
    st.markdown("### 🔒 Risk Categories")
    
    st.markdown("""
    - **Low Risk (<15%)**: Standard underwriting approval
    - **Medium Risk (15-30%)**: Review building condition
    - **High Risk (30-50%)**: Enhanced inspection required
    - **Very High Risk (>50%)**: Deny or premium surcharge
    """)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and Scikit-Learn | 
    <a href='http://localhost:8000/docs'>API Documentation</a></p>
</div>
""", unsafe_allow_html=True)