# 🏠 Insurance Claim Risk Prediction System

> Machine learning system for predicting insurance claim probability for buildings using structural, geographical, and usage characteristics.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Demo](https://via.placeholder.com/800x400.png?text=Add+Screenshot+Here)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## 🎯 Overview

This project develops a predictive model that estimates the probability of an insurance claim for insured buildings. The system supports insurance risk assessment and underwriting decisions by identifying key factors associated with claim occurrence.

### Business Impact

- **ROC-AUC Score**: 0.85 (significantly above baseline)
- **Recall**: 80% (high claim detection rate)
- **Real-world use**: Helps underwriters make data-driven decisions on policy approvals

---

## ✨ Features

### Core Functionality
- 🔮 **Claim Probability Prediction** - Returns probability score (0-1)
- 📊 **Risk Categorization** - Classifies buildings into Low/Medium/High/Very High risk
- 💡 **Underwriting Recommendations** - Actionable decisions for each risk level
- 🌍 **Geographic Risk Encoding** - Accounts for location-based risk factors

### User Interfaces
- 🖥️ **REST API** - FastAPI backend with automatic documentation
- 🎨 **Streamlit Web UI** - Interactive web interface for non-technical users
- 📱 **Responsive Design** - Works on desktop and mobile

### Technical Features
- ✅ Handles missing data (imputation + missingness flags)
- ✅ Feature engineering (15+ derived features)
- ✅ Class imbalance handling
- ✅ Anti-data-leakage (train/test isolation)
- ✅ Dockerized deployment
- ✅ Production-ready code structure

---

## 📈 Model Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.85 | Excellent discrimination |
| **Accuracy** | 82% | High overall correctness |
| **Precision** | 78% | Low false positive rate |
| **Recall** | 80% | Catches most real claims |
| **F1 Score** | 0.79 | Balanced performance |

### Key Predictors

1. **Geographic Location** (`Geo_Code`) - Highest importance
2. **Building Age** - Older buildings = higher risk
3. **Year of Observation** - Temporal trends matter
4. **Structural Risk Score** - Composite of maintenance factors
5. **Building Dimension** - Larger buildings = higher exposure

---

## 🛠️ Tech Stack

### Machine Learning
- **Scikit-Learn** - Model training (Gradient Boosting, Random Forest, Logistic Regression)
- **XGBoost / LightGBM** - Gradient boosting implementations
- **Pandas & NumPy** - Data processing
- **Joblib** - Model serialization

### Backend
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Frontend
- **Streamlit** - Web UI framework
- http://localhost:8504/

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **GitHub Actions** - CI/CD (coming soon)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- (Optional) Docker

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/insurance-claim-prediction.git
cd insurance-claim-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit UI
streamlit run app.py

# OR run FastAPI backend
uvicorn api.main:app --reload
```

### Using Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# Access applications
# - Streamlit UI: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

---

## 📚 API Documentation

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0"
}
```

#### `POST /predict`
Predict claim risk for a single building.

**Request:**
```json
{
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
```

**Response:**
```json
{
  "claim_probability": 0.2543,
  "risk_category": "Medium",
  "recommendation": "Review building age and structural condition",
  "model_version": "1.0"
}
```

#### `POST /batch_predict`
Predict risk for multiple buildings.

**Interactive API Documentation:** http://localhost:8000/docs

---
## 📁 Project Structure
