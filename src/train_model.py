# src/train_model.py
"""
Train script for Job Family classifier with Label Encoding for target.

Saves best pipeline into models/ as pipeline_rf.joblib or pipeline_xgb.joblib
Also saves LabelEncoder to models/label_encoder.joblib and metadata.json.
"""

import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

from preprocess import detect_columns, build_preprocessor, save_schema

# === CONFIG ===
DATA_PATH = "data/synthetic_students_with_skills_5000.csv"
TARGET = "job_family"
OUT_DIR = "models"
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df

def prepare_data(df):
    schema = detect_columns(df)
    save_schema(schema, os.path.join(OUT_DIR, "schema.json"))
    X = df[schema['feature_cols']].copy()
    y = df[TARGET].astype(str).copy()
    return X, y, schema

def encode_labels(y_train, y_test=None):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    if y_test is not None:
        y_test_enc = le.transform(y_test)
    else:
        y_test_enc = None
    return le, y_train_enc, y_test_enc

def build_and_train(X_train, y_train_enc, schema):
    # build preprocessor
    preprocessor = build_preprocessor(schema)
    # RandomForest pipeline
    rf_pipe = Pipeline(steps=[('preproc', preprocessor),
                              ('clf', RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE))])
    print("Training RandomForest...")
    rf_pipe.fit(X_train, y_train_enc)
    # XGBoost pipeline
    xgb_pipe = Pipeline(steps=[('preproc', preprocessor),
                               ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=4, random_state=RANDOM_STATE))])
    print("Training XGBoost...")
    xgb_pipe.fit(X_train, y_train_enc)
    return rf_pipe, xgb_pipe

def evaluate_and_select(best_out_dir, rf_pipe, xgb_pipe, X_test, y_test_enc, le):
    # Evaluate RF
    print("Evaluating RandomForest...")
    preds_rf_enc = rf_pipe.predict(X_test)
    probs_rf = rf_pipe.predict_proba(X_test)
    acc_rf = accuracy_score(y_test_enc, preds_rf_enc)
    top3_rf = top_k_accuracy_score(y_test_enc, probs_rf, k=3)
    print("RF acc: {:.4f}  top-3: {:.4f}".format(acc_rf, top3_rf))

    # Evaluate XGB
    print("Evaluating XGBoost...")
    preds_xgb_enc = xgb_pipe.predict(X_test)
    probs_xgb = xgb_pipe.predict_proba(X_test)
    acc_xgb = accuracy_score(y_test_enc, preds_xgb_enc)
    top3_xgb = top_k_accuracy_score(y_test_enc, probs_xgb, k=3)
    print("XGB acc: {:.4f} top-3: {:.4f}".format(acc_xgb, top3_xgb))

    # Choose best by top-3
    if top3_xgb >= top3_rf:
        chosen_pipe = xgb_pipe
        chosen_name = "pipeline_xgb.joblib"
        metrics = {"acc": float(acc_xgb), "top3": float(top3_xgb)}
        chosen_preds_enc = preds_xgb_enc
    else:
        chosen_pipe = rf_pipe
        chosen_name = "pipeline_rf.joblib"
        metrics = {"acc": float(acc_rf), "top3": float(top3_rf)}
        chosen_preds_enc = preds_rf_enc

    # Save chosen pipeline
    out_path = os.path.join(best_out_dir, chosen_name)
    joblib.dump(chosen_pipe, out_path)
    # Save label encoder
    le_path = os.path.join(best_out_dir, "label_encoder.joblib")
    joblib.dump(le, le_path)

    # For reporting, decode predictions and truth back to string labels
    decoded_preds = le.inverse_transform(chosen_preds_enc)
    # y_test_enc corresponds to encoded true labels; decode them
    decoded_y_true = le.inverse_transform(y_test_enc)

    # Create a classification report (strings)
    class_rep = classification_report(decoded_y_true, decoded_preds, zero_division=0, output_dict=True)

    meta = {
        "chosen_pipeline": chosen_name,
        "label_encoder": "label_encoder.joblib",
        "metrics": metrics,
        "model_details": "RandomForest/XGBoost pipeline, trained on synthetic data",
    }
    with open(os.path.join(best_out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved chosen pipeline to: {out_path}")
    print(f"Saved label encoder to: {le_path}")
    return chosen_pipe, class_rep, meta

def main():
    print("Loading data...")
    df = load_data()
    X, y, schema = prepare_data(df)
    # Stratified split on y (string labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)
    # Label encode targets
    le, y_train_enc, y_test_enc = encode_labels(y_train, y_test)
    # Train models on encoded labels
    rf_pipe, xgb_pipe = build_and_train(X_train, y_train_enc, schema)
    # Evaluate and select
    chosen_pipe, class_rep, meta = evaluate_and_select(OUT_DIR, rf_pipe, xgb_pipe, X_test, y_test_enc, le)
    # Print a short summary of classification report
    import pprint
    pp = pprint.PrettyPrinter(width=120)
    print("Sample of classification report for chosen model:")
    # print only top-level keys for brevity
    top_summary = {k: class_rep[k] for k in sorted(class_rep.keys())[:10]}
    pp.pprint(top_summary)
    print("Done.")

if __name__ == "__main__":
    main()
