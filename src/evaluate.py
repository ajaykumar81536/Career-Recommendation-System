# src/evaluate.py
"""
Evaluation helpers.

Functions:
- plot_confusion_matrix(y_true, y_pred, labels, out_path)
- top_k_accuracy(pipe, X_test, y_test, k)
- get_feature_names_from_pipeline(pipeline, schema)  # useful for feature importances
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

def plot_confusion_matrix(y_true, y_pred, labels=None, out_path="models/confusion.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title("Confusion matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    return out_path

def top_k_accuracy(pipe, X_test, y_test, k=3):
    probs = pipe.predict_proba(X_test)
    return top_k_accuracy_score(y_test, probs, k=k)

def get_feature_names_from_pipeline(pipeline, schema):
    """
    Attempt to reconstruct column names after the preprocessor.
    Works when categorical transformer exposes get_feature_names_out (sklearn >= 1.0).
    Returns list of feature names in order that the pipeline preprocessor will produce.
    """
    preproc = pipeline.named_steps.get('preproc', None)
    if preproc is None:
        raise ValueError("Pipeline does not contain 'preproc' named step.")
    feature_names = []
    # numeric cols
    num_cols = schema.get("numeric_cols", [])
    cat_cols = schema.get("categorical_cols", [])
    remainder_order = schema.get("binary_cols", []) + schema.get("interest_cols", []) + schema.get("skill_cols", [])
    # handle numeric
    if num_cols:
        feature_names.extend(num_cols)
    # handle categorical -> onehot
    if cat_cols:
        # The ColumnTransformer stores transformers in .transformers_
        for name, transformer, cols in preproc.transformers_:
            if name == 'cat':
                # transformer is a Pipeline with OneHotEncoder as last step
                ohe = transformer.named_steps.get('onehot', None)
                if ohe is not None:
                    try:
                        ohe_names = list(ohe.get_feature_names_out(cols))
                    except Exception:
                        # fallback: produce names like cat_col__value
                        ohe_names = []
                        for c in cols:
                            ohe_names.append(c)
                    feature_names.extend(ohe_names)
                else:
                    # fallback
                    feature_names.extend(cols)
    # remainder columns are passed through after transformed columns
    feature_names.extend(remainder_order)
    return feature_names
