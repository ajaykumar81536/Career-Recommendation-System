# src/preprocess.py
"""
Preprocessing utilities (compatible with multiple sklearn versions).

Functions:
- detect_columns(df): examine dataframe and return a schema dict listing feature columns.
- build_preprocessor(schema): returns a sklearn ColumnTransformer pipeline to preprocess data.
- save_schema(schema, path): save schema json
- load_schema(path): load schema json
"""

import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
# OneHotEncoder import only here; we'll instantiate carefully for version compatibility
from sklearn.preprocessing import OneHotEncoder

def detect_columns(df: pd.DataFrame):
    """
    Detect and return feature column groups from dataframe.
    Returns schema dict with keys:
      - feature_cols (ordered list used for X)
      - numeric_cols
      - categorical_cols
      - binary_cols
      - interest_cols
      - skill_cols
    """
    # heuristics -- adjust names if your columns differ
    skill_cols = [c for c in df.columns if c.startswith("skill__")]
    interest_cols = [c for c in df.columns if c.startswith("interest_")]
    # numeric candidates often present
    candidate_numeric = ["tenth_percent","twelfth_percent","current_cgpa","semester",
                         "internships_count","project_count","certifications_count",
                         "communication_skill_score","problem_solving_score"]
    numeric_cols = [c for c in candidate_numeric if c in df.columns]
    categorical_candidates = ["department","preferred_company_type"]
    categorical_cols = [c for c in categorical_candidates if c in df.columns]
    binary_candidates = ["strong_programming","strong_maths","strong_dsa",
                         "open_source_contrib","leadership_roles","pref_higher_studies",
                         "pref_immediate_job","pref_relocate"]
    binary_cols = [c for c in binary_candidates if c in df.columns]
    # final feature order: numeric, categorical, binary, interest, skill
    feature_cols = numeric_cols + categorical_cols + binary_cols + interest_cols + skill_cols
    schema = {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "binary_cols": binary_cols,
        "interest_cols": interest_cols,
        "skill_cols": skill_cols
    }
    return schema

def _make_onehot_encoder():
    """
    Instantiate OneHotEncoder compatible with both old and new sklearn versions.
    New versions use 'sparse_output' param; older used 'sparse'.
    We'll try sparse_output first, then fallback to sparse.
    """
    try:
        # try new param name
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        # fallback for older sklearn that expects 'sparse'
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    return ohe

def build_preprocessor(schema: dict):
    """
    Build a ColumnTransformer that:
     - scales numeric columns (StandardScaler)
     - one-hot-encodes categorical columns (OneHotEncoder with compatibility)
     - passes through binary + interest + skill multi-hot columns
    Returns the ColumnTransformer instance ready to be used inside a Pipeline.
    """
    numeric_cols = schema.get("numeric_cols", [])
    categorical_cols = schema.get("categorical_cols", [])
    # numeric transformer
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    # categorical transformer
    categorical_transformer = Pipeline(steps=[('onehot', _make_onehot_encoder())])
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    # remainder='passthrough' will keep binary, interest, skill columns in original order after transformed ones
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return preprocessor

def save_schema(schema: dict, path="models/schema.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)

def load_schema(path="models/schema.json"):
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # quick smoke test when run directly (adjust path if running from src/)
    import pandas as pd
    sample_path = os.path.join("..", "data", "synthetic_students_with_skills_5000.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        schema = detect_columns(df)
        print("Detected schema:")
        print(json.dumps(schema, indent=2))
        pre = build_preprocessor(schema)
        print("Preprocessor built:", pre)
    else:
        print("No sample dataset found at", sample_path)
