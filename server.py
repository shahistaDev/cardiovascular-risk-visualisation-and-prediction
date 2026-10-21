import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CardioRisk 3D Prediction & Explainability API",
    description="Track A Backend Service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved bundle containing end-to-end pipelines and preprocessor
print("Loading model pipelines and explainer...")
bundle = joblib.load("cardio_models.joblib")
pipelines = bundle["pipelines"]
preprocessor = bundle["preprocessor"]
explainer = bundle["explainer"]
raw_feature_names = bundle["raw_feature_names"]
processed_feature_names = bundle["processed_feature_names"]
print("All pipelines loaded successfully!")

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "CardioRisk Inference Engine"}

@app.post("/predict")
def predict_patient(patient_data: dict):
    try:
        # Convert dictionary to DataFrame
        df_in = pd.DataFrame([patient_data])

        # Ensure exact column presence and order as seen during training
        for col in raw_feature_names:
            if col not in df_in.columns:
                df_in[col] = None
        df_in = df_in[raw_feature_names]

        # Clean string values (title case and whitespace stripping) to match training categories
        cat_cols = [c for c in df_in.columns if not pd.api.types.is_numeric_dtype(df_in[c])]
        for c in cat_cols:
            df_in[c] = df_in[c].astype(str).str.strip().str.title()

        # Run inference directly through each pipeline
        cad_risk = float(pipelines["CAD"].predict_proba(df_in)[0, 1])
        lad_risk = float(pipelines["LAD"].predict_proba(df_in)[0, 1])
        lcx_risk = float(pipelines["LCX"].predict_proba(df_in)[0, 1])
        rca_risk = float(pipelines["RCA"].predict_proba(df_in)[0, 1])

        # Compute local SHAP attributions for clinical interpretability
        X_proc = preprocessor.transform(df_in)
        shap_vals = explainer.shap_values(X_proc)

        # Account for binary classification SHAP shapes
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

        return {
            "overall_cad_risk": round(cad_risk, 3),
            "vessels": {
                "LAD": round(lad_risk, 3),
                "LCX": round(lcx_risk, 3),
                "RCA": round(rca_risk, 3)
            },
            "top_factors": explanations[:5],
            "disclaimer": "Predictions are for clinical decision support / educational purposes only and not a substitute for formal diagnostic imaging."
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
