import joblib
import pandas as pd
import numpy as np

print("Testing cardio_models.joblib with server.py inference logic...")
bundle = joblib.load("cardio_models.joblib")

pipelines = bundle["pipelines"]
preprocessor = bundle["preprocessor"]
explainer = bundle["explainer"]
raw_feature_names = bundle["raw_feature_names"]
processed_feature_names = bundle["processed_feature_names"]

print(f"Pipelines loaded: {list(pipelines.keys())}")
print(f"Raw features count: {len(raw_feature_names)}")
print(f"Processed features count: {len(processed_feature_names)}")

# Load a sample patient
df = pd.read_csv("cad_dataset_clean.csv")
sample_dict = df[raw_feature_names].iloc[0].to_dict()

# Simulate server.py /predict
df_in = pd.DataFrame([sample_dict])
for col in raw_feature_names:
    if col not in df_in.columns:
        df_in[col] = None
df_in = df_in[raw_feature_names]

cat_cols = [c for c in df_in.columns if not pd.api.types.is_numeric_dtype(df_in[c])]
for c in cat_cols:
    df_in[c] = df_in[c].astype(str).str.strip().str.title()

cad_risk = float(pipelines["CAD"].predict_proba(df_in)[0, 1])
lad_risk = float(pipelines["LAD"].predict_proba(df_in)[0, 1])
lcx_risk = float(pipelines["LCX"].predict_proba(df_in)[0, 1])
rca_risk = float(pipelines["RCA"].predict_proba(df_in)[0, 1])

X_proc = preprocessor.transform(df_in)
shap_vals = explainer.shap_values(X_proc)

if isinstance(shap_vals, list):
    sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
elif hasattr(shap_vals, "shape") and len(shap_vals.shape) == 3:
    sv = shap_vals[0, :, 1]
elif hasattr(shap_vals, "shape") and len(shap_vals.shape) == 2:
    sv = shap_vals[0]
else:
    sv = np.array(shap_vals).flatten()

explanations = []
for name, imp in zip(processed_feature_names, sv):
    explanations.append({
        "feature": str(name),
        "impact": round(float(imp), 3),
        "effect": "Increases Risk" if float(imp) > 0 else "Lowers Risk"
    })
explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)

result = {
    "overall_cad_risk": round(cad_risk, 3),
    "vessels": {
        "LAD": round(lad_risk, 3),
        "LCX": round(lcx_risk, 3),
        "RCA": round(rca_risk, 3)
    },
    "top_factors": explanations[:5]
}

print("\nServer inference test SUCCESS!")
print(f"Overall CAD Risk: {result['overall_cad_risk']}")
print(f"Vessel Risks: {result['vessels']}")
print("Top 5 SHAP Clinical Factors:")
for f in result['top_factors']:
    print(f"  • {f['feature']}: {f['impact']} ({f['effect']})")

