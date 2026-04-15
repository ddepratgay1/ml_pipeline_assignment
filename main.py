# ============================================================
# main.py
# Assignment: Automated ML Pipelines & Model Serving
# Part 3: Serving the trained model as a REST API
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.classification import load_model, predict_model

# Create the FastAPI app instance
app = FastAPI(title="Paddy Variety Predictor API")

# Load the saved PyCaret pipeline once when the server starts.
# This avoids reloading the model on every request (slow).
pipeline = load_model('best_pipeline')

# ── Define the input schema ───────────────────────────────────
# Pydantic's BaseModel validates that incoming JSON has the right fields.
# These should match your dataset's feature columns (minus Variety and Yield).
# Below is a representative subset — include all 43 feature columns for full accuracy.
class PaddyInput(BaseModel):
    Hectares: float
    Agriblock: str
    Soil_Types: str          # note: FastAPI doesn't allow spaces in field names
    Seedrate: float
    LP_Mainfield: float
    Nursery: str
    Nursery_area: float
    LP_nurseryarea: float
    DAP_20days: float
    Weed28D_thiobencarb: float
    Urea_40Days: float
    Potassh_50Days: float
    Micronutrients_70Days: float
    Pest_60Day: float
    Rain_D1_30: float
    DAI_D1_30: float
    Rain_D30_50: float
    DAI_D30_50: float
    Rain_D51_70: float
    AI_D51_70: float
    Rain_D71_105: float
    DAI_D71_105: float
    Min_temp_D1_D30: float
    Max_temp_D1_D30: float
    Min_temp_D31_D60: float
    Max_temp_D31_D60: float
    Min_temp_D61_D90: float
    Max_temp_D61_D90: float
    Min_temp_D91_D120: float
    Max_temp_D91_D120: float
    Wind_D1_D30: float
    Wind_D31_D60: float
    Wind_D61_D90: float
    Wind_D91_D120: float
    WindDir_D1_D30: str
    WindDir_D31_D60: str
    WindDir_D61_D90: str
    WindDir_D91_D120: str
    Humidity_D1_D30: float
    Humidity_D31_D60: float
    Humidity_D61_D90: float
    Humidity_D91_D120: float
    Trash: float

# ── Map friendly field names back to original CSV column names ─
def input_to_dataframe(data: PaddyInput) -> pd.DataFrame:
    """Convert API input to a DataFrame matching the training data format."""
    row = {
        'Hectares': data.Hectares,
        'Agriblock': data.Agriblock,
        'Soil Types': data.Soil_Types,
        'Seedrate(in Kg)': data.Seedrate,
        'LP_Mainfield(in Tonnes)': data.LP_Mainfield,
        'Nursery': data.Nursery,
        'Nursery area (Cents)': data.Nursery_area,
        'LP_nurseryarea(in Tonnes)': data.LP_nurseryarea,
        'DAP_20days': data.DAP_20days,
        'Weed28D_thiobencarb': data.Weed28D_thiobencarb,
        'Urea_40Days': data.Urea_40Days,
        'Potassh_50Days': data.Potassh_50Days,
        'Micronutrients_70Days': data.Micronutrients_70Days,
        'Pest_60Day(in ml)': data.Pest_60Day,
        '30DRain( in mm)': data.Rain_D1_30,
        '30DAI(in mm)': data.DAI_D1_30,
        '30_50DRain( in mm)': data.Rain_D30_50,
        '30_50DAI(in mm)': data.DAI_D30_50,
        '51_70DRain(in mm)': data.Rain_D51_70,
        '51_70AI(in mm)': data.AI_D51_70,
        '71_105DRain(in mm)': data.Rain_D71_105,
        '71_105DAI(in mm)': data.DAI_D71_105,
        'Min temp_D1_D30': data.Min_temp_D1_D30,
        'Max temp_D1_D30': data.Max_temp_D1_D30,
        'Min temp_D31_D60': data.Min_temp_D31_D60,
        'Max temp_D31_D60': data.Max_temp_D31_D60,
        'Min temp_D61_D90': data.Min_temp_D61_D90,
        'Max temp_D61_D90': data.Max_temp_D61_D90,
        'Min temp_D91_D120': data.Min_temp_D91_D120,
        'Max temp_D91_D120': data.Max_temp_D91_D120,
        'Inst Wind Speed_D1_D30(in Knots)': data.Wind_D1_D30,
        'Inst Wind Speed_D31_D60(in Knots)': data.Wind_D31_D60,
        'Inst Wind Speed_D61_D90(in Knots)': data.Wind_D61_D90,
        'Inst Wind Speed_D91_D120(in Knots)': data.Wind_D91_D120,
        'Wind Direction_D1_D30': data.WindDir_D1_D30,
        'Wind Direction_D31_D60': data.WindDir_D31_D60,
        'Wind Direction_D61_D90': data.WindDir_D61_D90,
        'Wind Direction_D91_D120': data.WindDir_D91_D120,
        'Relative Humidity_D1_D30': data.Humidity_D1_D30,
        'Relative Humidity_D31_D60': data.Humidity_D31_D60,
        'Relative Humidity_D61_D90': data.Humidity_D61_D90,
        'Relative Humidity_D91_D120': data.Humidity_D91_D120,
        'Trash(in bundles)': data.Trash,
    }
    return pd.DataFrame([row])

# ── The prediction endpoint ───────────────────────────────────
@app.post("/predict")
def predict(data: PaddyInput):
    """
    Accepts farming condition data as JSON.
    Returns the predicted paddy variety.
    """
    input_df = input_to_dataframe(data)
    predictions = predict_model(pipeline, data=input_df)
    predicted_class = predictions['prediction_label'].iloc[0]
    predicted_score = round(float(predictions['prediction_score'].iloc[0]), 4)

    return {
        "predicted_variety": predicted_class,
        "confidence": predicted_score
    }

# ── Health check endpoint ─────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Paddy Variety Predictor API is running 🌾"}