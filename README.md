# Paddy Variety ML Pipeline

## Overview
This project builds an automated ML pipeline to classify paddy (rice) varieties
based on farming and environmental conditions. The dataset contains 2,790 rows
and 44 features sourced from the UC Irvine Machine Learning Repository. The
target variable is `Variety`, which has 3 classes: CO_43, ponmani, and delux ponni.

## Files
- `discovery.py` — PyCaret vs Scikit-Learn model comparison and evaluation
- `main.py` — FastAPI model serving application
- `paddydataset.csv` — Source dataset
- `best_pipeline.pkl` — Saved PyCaret model pipeline
- `report.docx` — Screenshots and evaluation results
- `README.md` — Project documentation

## How to Run

### 1. Create and activate the Conda environment
```bash
conda create -n ml_pipeline2 python=3.11 -y
conda activate ml_pipeline2
```

### 2. Install dependencies
```bash
conda install -c conda-forge pycaret -y
pip install fastapi uvicorn python-multipart
```

### 3. Run model comparison
```bash
python discovery.py
```

### 4. Start the API server
```bash
python -m uvicorn main:app --reload
```
Then open your browser and go to: http://127.0.0.1:8000/docs

## Results
PyCaret's `compare_models()` tested 14 algorithms automatically. The best
model was **Logistic Regression** with **100% accuracy** on the paddy variety
classification task. Four other models also achieved 100% accuracy (Decision
Tree, QDA, Gradient Boosting, LightGBM), suggesting the features in this
dataset are highly separable by variety.

The manual Scikit-Learn Random Forest achieved **99% accuracy**, slightly lower
due to using a single train-test split rather than cross-validation.

## Sample API Input/Output

**Endpoint:** `POST /predict`

**Input:**
```json
{
  "Hectares": 6,
  "Agriblock": "Cuddalore",
  "Soil_Types": "alluvial",
  "Nursery": "dry",
  "WindDir_D1_D30": "SW",
  "Humidity_D1_D30": 72,
  "..."  : "..."
}
```

**Output:**
```json
{
  "predicted_variety": "CO_43",
  "confidence": 1.0
}
```

## Discussion
PyCaret proved significantly more efficient than the manual Scikit-Learn workflow.
With just a few function calls, PyCaret automated preprocessing, model training,
cross-validation, and comparison across 14 algorithms. The manual approach required
explicit encoding, scaling, splitting, and training — more code but more transparency.
Results differed slightly because PyCaret uses stratified k-fold cross-validation
while the manual approach used a single 80/20 split.