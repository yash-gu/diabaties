# Pima Diabetes FastAPI Demo

## Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Train model
python3 train.py  # creates model.pkl

## Run API
uvicorn app.main:app --reload  # http://localhost:8000/docs

## Test
python3 score.py
# or curl POST to /predict

