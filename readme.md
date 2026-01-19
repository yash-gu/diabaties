cat > README.md << 'EOF'
# 🏥 Diabetes Risk Prediction - MLOps Project

A complete end-to-end Machine Learning Operations (MLOps) project that demonstrates ML model development, API deployment, CI/CD automation, Docker containerization, and UI/UX design.

## 🎯 Project Overview

This project predicts diabetes risk using a machine learning model trained on the PIMA Indians Diabetes Dataset. It features:
- **ML Model**: Random Forest classifier with 74.7% test accuracy
- **REST API**: FastAPI backend for predictions
- **Web UI**: Beautiful, responsive frontend with real-time predictions
- **CI/CD Pipeline**: GitHub Actions for automated testing and Docker image building
- **Containerization**: Docker image pushed to AWS ECR
- **Deployment Ready**: Can be deployed to AWS EC2 or EKS

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────┐
│ User Interface (Web) │
│ HTML/CSS/JavaScript Frontend │
│ │
│ ┌──────────────────────────────────────────────────┐ │
│ │ Form Inputs → Predict Button → Result Card │ │
│ │ (Beautiful Purple Gradient Theme) │ │
│ └──────────────────────────────────────────────────┘ │
└─────────────────────┬──────────────────────────────────┘
│ HTTP POST /predict
▼
┌─────────────────────────────────────────────────────────┐
│ FastAPI REST API Backend │
│ (app/main.py - Uvicorn Server) │
│ │
│ ┌──────────────────────────────────────────────────┐ │
│ │ /predict → Load model.pkl → Run inference │ │
│ │ /health → Health check endpoint │ │
│ │ / → Serve HTML frontend │ │
│ └──────────────────────────────────────────────────┘ │
└─────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│ ML Model (Random Forest Classifier) │
│ model.pkl (3.4 MB) │
│ Trained on PIMA Dataset │
│ Test Accuracy: 74.68% │
└─────────────────────────────────────────────────────────┘


## 🚀 Features

### Machine Learning
- **Dataset**: PIMA Indians Diabetes Dataset (768 samples, 8 features)
- **Model**: Random Forest Classifier
- **Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Accuracy**: 74.68% on test set

### Backend API
- **Framework**: FastAPI (Python)
- **Server**: Uvicorn
- **Endpoints**:
  - `GET /` - Serve web UI
  - `POST /predict` - Get diabetes risk prediction
  - `GET /health` - Health check
  - `GET /docs` - Swagger API documentation

### Frontend UI/UX
- **Framework**: Vanilla HTML/CSS/JavaScript
- **Design**: Modern gradient theme (purple/violet)
- **Features**:
  - Real-time form validation
  - Loading spinner animation
  - Color-coded results (green = low risk, red = high risk)
  - Responsive mobile design
  - Helpful tooltips for each input

### DevOps & CI/CD
- **Version Control**: GitHub
- **CI/CD**: GitHub Actions workflow
- **Containerization**: Docker & Docker Compose
- **Registry**: AWS ECR (Elastic Container Registry)
- **Testing**: pytest for unit tests
- **Infrastructure**: Ready for AWS EC2/EKS deployment

## 📋 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- Git

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/yash-gu/diabaties.git
cd diabaties

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (if needed)
python3 train.py

# 5. Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Open in browser
# http://localhost:8000
