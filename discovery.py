# ============================================================
# discovery.py
# Assignment: Automated ML Pipelines & Model Serving
# Part 2: Comparing PyCaret vs Scikit-Learn workflows
# ============================================================

import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # suppresses annoying but harmless warnings

# ── Load the dataset ─────────────────────────────────────────
# pd.read_csv reads the CSV file into a DataFrame (like a table in Python)
df = pd.read_csv('paddydataset.csv')

# Strip whitespace from column names (the CSV has trailing spaces)
df.columns = df.columns.str.strip()

# Our TARGET is 'Variety' — the 3 rice types we want to classify
# We drop 'Paddy yield(in Kg)' because it's a numeric outcome, not a feature
df = df.drop(columns=['Paddy yield(in Kg)'])

print("Dataset shape:", df.shape)           # should print (2790, 44)
print("Target classes:", df['Variety'].unique())  # CO_43, ponmani, delux ponni


# ============================================================
# PART A: PyCaret Workflow (Low-Code)
# ============================================================
# PyCaret automates all the messy ML steps: encoding, scaling,
# splitting, training, and comparing models — with just a few lines.

from pycaret.classification import (
    setup, compare_models, plot_model, save_model, pull
)

# setup() prepares the data for ML.
# - target='Variety' tells PyCaret what column we're predicting
# - session_id=42 sets a random seed so results are reproducible
# - verbose=False keeps the output clean
print("\n--- Setting up PyCaret ---")
clf_setup = setup(
    data=df,
    target='Variety',
    session_id=42,
    verbose=False
)

# compare_models() trains ~15 different algorithms and ranks them by accuracy.
# n_select=3 keeps the top 3.
# This is the magic of PyCaret — one line replaces hundreds of lines of code.
print("\n--- Comparing models (this may take 1-2 minutes) ---")
top3_models = compare_models(n_select=3, verbose=True)

# The best model is always the first in the list
best_model = top3_models[0]
print("\n✅ Best Model:", type(best_model).__name__)

# Pull the comparison table as a DataFrame and print it
comparison_table = pull()
print("\n--- Model Comparison Table ---")
print(comparison_table)

# plot_model() generates a Confusion Matrix for the best model.
# A confusion matrix shows how many predictions were correct vs wrong,
# broken down by class. save=True saves it as a PNG file.
print("\n--- Generating Confusion Matrix ---")
plot_model(best_model, plot='confusion_matrix', save=True)
print("✅ Confusion matrix saved as 'Confusion Matrix.png'")

# Save the full trained pipeline (preprocessing + model) to a file.
# This is what we'll load later in FastAPI to make predictions.
save_model(best_model, 'best_pipeline')
print("✅ Model saved as 'best_pipeline.pkl'")


# ============================================================
# PART B: Scikit-Learn Workflow (Manual)
# ============================================================
# Here we replicate what PyCaret did automatically, but by hand.
# This is more code but gives you full control.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier   # change this to match your best model
from sklearn.metrics import classification_report
import numpy as np

print("\n\n--- Scikit-Learn Manual Workflow ---")

# Reload a fresh copy of the data
df2 = pd.read_csv('paddydataset.csv')
df2.columns = df2.columns.str.strip()
df2 = df2.drop(columns=['Paddy yield(in Kg)'])

# -- STEP 1: Encode categorical features --
# ML models only understand numbers, not strings like "clay" or "SW"
# LabelEncoder converts each unique category to a number (e.g. clay=0, alluvial=1)
le = LabelEncoder()
categorical_cols = df2.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Variety')  # Don't encode the target yet

for col in categorical_cols:
    df2[col] = le.fit_transform(df2[col].astype(str))

# -- STEP 2: Separate features (X) from target (y) --
X = df2.drop(columns=['Variety'])
y = le.fit_transform(df2['Variety'])  # encode target too: 0, 1, or 2

# -- STEP 3: Scale numeric features --
# StandardScaler transforms each feature to have mean=0 and std=1.
# This prevents features with large values (like rainfall in mm)
# from dominating features with small values (like temperature).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -- STEP 4: Split into training and test sets --
# 80% of rows go to training, 20% are held out for testing.
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

# -- STEP 5: Train the model --
# We use Random Forest (a common best performer). If PyCaret's best
# model was different (e.g. LightGBM), you'd swap this out.
model_sklearn = RandomForestClassifier(n_estimators=100, random_state=42)
model_sklearn.fit(X_train, y_train)

# -- STEP 6: Evaluate --
# classification_report shows precision, recall, and F1 score per class.
y_pred = model_sklearn.predict(X_test)
print("\n--- Classification Report (Scikit-Learn) ---")
print(classification_report(y_pred, y_test))


# ============================================================
# SYNTHESIS COMMENT (200 words)
# ============================================================
#
# Workflow Efficiency Comparison:
#
# The PyCaret workflow was significantly more efficient than the manual
# Scikit-Learn approach. PyCaret compressed what would be hundreds of lines
# of code — data preprocessing, model training, cross-validation, and
# performance comparison across ~15 algorithms — into just a handful of
# function calls (setup, compare_models, plot_model). This drastically
# reduces development time and the risk of implementation errors.
#
# The manual Scikit-Learn workflow required explicit handling of every step:
# encoding categoricals, scaling numerics, splitting the data, choosing a
# model, training, and evaluating. While more verbose, this approach offers
# deeper transparency and control — you understand exactly what transformations
# are applied and can fine-tune each step independently.
#
# Results may differ slightly between the two workflows because PyCaret uses
# stratified k-fold cross-validation internally (typically 10 folds), while
# the manual approach uses a single 80/20 train-test split. Cross-validation
# averages performance across multiple folds, producing more stable and
# slightly different accuracy estimates. Additionally, PyCaret applies its
# own preprocessing pipeline (e.g., handling rare categories, normalization
# strategies) that may differ from the manual StandardScaler + LabelEncoder
# combination used here. For rapid prototyping, PyCaret wins. For production
# fine-tuning, Scikit-Learn gives more control.
#