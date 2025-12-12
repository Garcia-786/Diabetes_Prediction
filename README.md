# Diabetes Prediction

Simple Streamlit app that trains a RandomForest model on a diabetes dataset and provides an interactive UI to inspect the data and make predictions.

## What’s included
- `diabetes.py` — Streamlit app that loads the dataset, trains the model, shows plots, and predicts on user input.

## Setup
1. Create a virtual environment (recommended):

```bash
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the dataset `diabetes - diabetes.csv` in the project root. The app currently expects the file at `C:\Diabetes Prediction\diabetes - diabetes.csv` — you can update the path in `diabetes.py` if needed.

## Run

```bash
streamlit run diabetes.py
```

## Notes
- Dataset is not committed by default (it's included in `.gitignore`). If you want to publish a small sample dataset, add it intentionally.
- See [diabetes.py](diabetes.py) for implementation details.
