# app/ui_streamlit.py

# Polished Streamlit UI (section-wise inputs) + donut chart + robust dtype handling.

# Run:
#    python -m streamlit run app\ui_streamlit.py


import streamlit as st
from pathlib import Path
import joblib, json, pandas as pd, numpy as np, os
import plotly.express as px

ROOT = Path(".")
MODEL_XGB = ROOT / "models" / "pipeline_xgb.joblib"
MODEL_RF  = ROOT / "models" / "pipeline_rf.joblib"
LE_PATH   = ROOT / "models" / "label_encoder.joblib"
SCHEMA    = ROOT / "models" / "schema.json"
SKILL_LIST = ROOT / "data" / "extracted_skill_list.csv"
ROLE_SKILLS = ROOT / "data" / "job_role_to_skills.csv"

# === Load artifacts (exit if missing) ===
if MODEL_XGB.exists():
    PIPELINE_PATH = MODEL_XGB
elif MODEL_RF.exists():
    PIPELINE_PATH = MODEL_RF
else:
    raise SystemExit("No trained pipeline found in models/. Run training first.")

if not LE_PATH.exists():
    raise SystemExit("Label encoder not found at models/label_encoder.joblib. Run training first.")
if not SCHEMA.exists():
    raise SystemExit("Schema not found at models/schema.json. Run preprocess/save_schema during training.")

pipe = joblib.load(PIPELINE_PATH)
le = joblib.load(LE_PATH)
schema = json.load(open(SCHEMA, "r"))

feature_cols = schema["feature_cols"]
numeric_cols = schema.get("numeric_cols", [])
categorical_cols = schema.get("categorical_cols", [])
binary_cols = schema.get("binary_cols", [])
interest_cols = schema.get("interest_cols", [])
skill_cols = schema.get("skill_cols", [])

# load lists
skills_all = list(pd.read_csv(SKILL_LIST)['skill'].astype(str).tolist()) if SKILL_LIST.exists() else []
role_skill_df = pd.read_csv(ROLE_SKILLS) if ROLE_SKILLS.exists() else pd.DataFrame()

# helper functions
def rank_roles_by_skill_overlap(selected_skills, top_k=5):
    if role_skill_df.empty:
        return []
    candidates = []
    for _, row in role_skill_df.iterrows():
        role = row['job_role']
        skills = [s.strip() for s in str(row.get('skills','')).split("|") if s.strip()]
        overlap = len(set(skills).intersection(set(selected_skills)))
        score = overlap / (len(skills)+1e-9)
        candidates.append((score, overlap, role, skills))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = []
    for score, overlap, role, skills in candidates[:top_k]:
        out.append({"role": role, "score": float(score), "overlap_count": int(overlap), "required_skills": skills})
    return out

def compute_skill_gaps(required_skills, selected_skills, top_n=8):
    req = [s for s in required_skills if isinstance(s, str) and s.strip()]
    missing = [s for s in req if s not in selected_skills]
    return missing[:top_n]

# UI layout
st.set_page_config(page_title="AI Career Path Recommender", layout="wide")
st.title("AI Career Path Recommender 🚀")

st.markdown("""
Enter your profile section-wise. The model recommends top-3 job families (emphasizes Data Science/AI & Software Dev).
""")

with st.expander("1) Academics (required)", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        current_cgpa = st.number_input("Current CGPA (0-10)", min_value=0.0, max_value=10.0, value=7.2, step=0.01)
        semester = st.selectbox("Semester", options=list(range(2,9)), index=3)
    with col2:
        tenth = st.number_input("10th %", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        twelfth = st.number_input("12th % / Diploma %", min_value=0.0, max_value=100.0, value=72.0, step=0.1)
    with col3:
        department = st.selectbox("Department", options=["Computer Science & Engineering","Electronics & Communication","Mechanical Engineering","Civil Engineering","Biotechnology Engineering"], index=0)
        preferred_company = st.selectbox("Preferred company type", options=["Startup","MNC","PSU","Research"], index=1)

with st.expander("2) Interests & Strengths", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        interest_ai_ml = st.checkbox("Interest: AI / Machine Learning", value=True)
        interest_web = st.checkbox("Interest: Web / Frontend / Backend", value=False)
        interest_mobile = st.checkbox("Interest: Mobile / App", value=False)
        interest_cloud = st.checkbox("Interest: Cloud / DevOps", value=False)
    with col2:
        strong_programming = st.radio("Programming skill", ["No","Yes"], index=1)
        strong_maths = st.radio("Mathematics strength", ["No","Yes"], index=1)
        strong_dsa = st.radio("DSA strength", ["No","Yes"], index=1)

with st.expander("3) Experience & Extracurriculars", expanded=False):
    internships = st.number_input("No. of internships", min_value=0, max_value=10, value=1)
    projects = st.number_input("No. of projects", min_value=0, max_value=20, value=3)
    open_source = st.checkbox("Open-source contributions", value=False)
    hackathons = st.number_input("Hackathons attended", min_value=0, max_value=20, value=0)
    leadership = st.checkbox("Held leadership roles", value=False)
    certifications = st.number_input("No. of certifications", min_value=0, max_value=50, value=1)

with st.expander("4) Skills (search & select)", expanded=True):
    selected_skills = st.multiselect("Pick your skills", skills_all, default=["Python"] if "Python" in skills_all else [])

with st.form("form_recommend"):
    pref_immediate_job = st.checkbox("Prefer immediate job", value=True)
    pref_higher_studies = st.checkbox("Prefer higher studies", value=False)
    pref_relocate = st.checkbox("Willing to relocate", value=True)
    submit = st.form_submit_button("Recommend")

if submit:
    # build input row with schema order & defaults
    row = {}
    for c in feature_cols:
        if c in categorical_cols:
            row[c] = ""
        else:
            row[c] = 0

    # fill numbers & cats
    if "current_cgpa" in feature_cols:
        row["current_cgpa"] = float(current_cgpa)
    if "semester" in feature_cols:
        row["semester"] = int(semester)
    if "tenth_percent" in feature_cols:
        row["tenth_percent"] = float(tenth)
    if "twelfth_percent" in feature_cols:
        row["twelfth_percent"] = float(twelfth)
    if "department" in feature_cols:
        row["department"] = str(department)
    if "preferred_company_type" in feature_cols:
        row["preferred_company_type"] = str(preferred_company)

    # binary flags
    for name, val in [("pref_immediate_job", pref_immediate_job), ("pref_higher_studies", pref_higher_studies), ("pref_relocate", pref_relocate)]:
        if name in feature_cols:
            row[name] = 1 if val else 0
    for name, val in [("strong_programming", strong_programming == "Yes"), ("strong_maths", strong_maths == "Yes"), ("strong_dsa", strong_dsa == "Yes")]:
        if name in feature_cols:
            row[name] = 1 if val else 0
    if "internships_count" in feature_cols:
        row["internships_count"] = int(internships)
    if "project_count" in feature_cols:
        row["project_count"] = int(projects)
    if "open_source_contrib" in feature_cols:
        row["open_source_contrib"] = 1 if open_source else 0
    if "leadership_roles" in feature_cols:
        row["leadership_roles"] = 1 if leadership else 0
    if "certifications_count" in feature_cols:
        row["certifications_count"] = int(certifications)
    if "interest_ai_ml" in feature_cols:
        row["interest_ai_ml"] = 1 if interest_ai_ml else 0
    # set other interest flags if present
    if "interest_web_dev" in feature_cols:
        row["interest_web_dev"] = 1 if interest_web else 0
    if "interest_app_dev" in feature_cols:
        row["interest_app_dev"] = 1 if interest_mobile else 0
    if "interest_cloud_devops" in feature_cols:
        row["interest_cloud_devops"] = 1 if interest_cloud else 0

    # skills
    for s in selected_skills:
        col = f"skill__{s}"
        if col in feature_cols:
            row[col] = 1

    # ensure fields exist and cast types
    X = pd.DataFrame([row], columns=feature_cols)
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("")
    remaining = [c for c in feature_cols if c not in numeric_cols + categorical_cols]
    for c in remaining:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)

    # Predict
    probs = pipe.predict_proba(X)[0]
    # ensure python floats
    probs = [float(p) for p in probs]
    class_order = list(le.classes_)
    topk = 3
    topk_idx = np.argsort(probs)[-topk:][::-1]
    topk_labels = le.inverse_transform(topk_idx)
    topk_scores = [probs[i] for i in topk_idx]

    # Winner card + donut chart
    winner = topk_labels[0]
    st.markdown("## Recommendation")
    winner_col, chart_col = st.columns([2,1])
    with winner_col:
        st.markdown(f"### 🔹 Top recommendation — **{winner.replace('_',' ').title()}**")
        st.markdown(f"**Confidence:** {topk_scores[0]:.3f}")
        # Highlight if DS/AI or Software Dev
        if winner in ("DATA_SCIENCE/AI","SOFTWARE_DEV"):
            st.success(f"This looks like a strong fit for {winner.replace('_',' ').title()} based on selected skills & interests.")
        # short reason heuristic
        reasons = []
        if row.get("interest_ai_ml",0): reasons.append("Interest in AI/ML")
        if row.get("strong_programming",0): reasons.append("Strong programming")
        if row.get("internships_count",0)>0: reasons.append(f"{row['internships_count']} internship(s)")
        st.write(", ".join(reasons))

    with chart_col:
        df_chart = pd.DataFrame({
            "family": [lbl.replace("_"," ").title() for lbl in topk_labels],
            "score": [float(s) for s in topk_scores]
        })
        fig = px.pie(df_chart, names="family", values="score", hole=0.55,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart with percentages
    st.markdown("### Top 3 Families (detailed)")
    bar_df = pd.DataFrame({"family":[lbl.replace("_"," ").title() for lbl in topk_labels],"score":[float(s) for s in topk_scores]})
    st.bar_chart(bar_df.set_index("family"))

    # Show top roles inside winner (skill-overlap)
    st.markdown("---")
    st.subheader(f"Top roles inside {winner.replace('_',' ').title()}")
    ranked_roles = rank_roles_by_skill_overlap(selected_skills, top_k=6)
    if not ranked_roles:
        st.write("Role-level mapping not available.")
    else:
        for i, r in enumerate(ranked_roles[:3], start=1):
            st.markdown(f"**{i}. {r['role']}** — match score {r['score']:.2f} (matched {r['overlap_count']} skills)")
            if r['required_skills']:
                missing = compute_skill_gaps(r['required_skills'], selected_skills, top_n=6)
                st.write("Top required skills (sample): " + ", ".join(r['required_skills'][:8]))
                if missing:
                    st.warning("Skill gap: " + ", ".join(missing[:5]))
                else:
                    st.success("No major skill gap detected for this role.")

    st.markdown("---")
    st.info("Tip: add more skills (multi-select). For production, we'll combine ML + rule engine weighting to return refined ranked roles.")
