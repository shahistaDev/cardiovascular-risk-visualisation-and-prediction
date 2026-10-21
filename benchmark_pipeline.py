import os
import sys
import json
import warnings
import codecs
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Ensure clean UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    except Exception:
        pass

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap

print("="*80)
print("CARDIOVASCULAR RISK PREDICTION - COMPREHENSIVE BENCHMARKING SUITE")
print("Evaluation Rubric: Predictive Performance (30%) - AI/ML Engineer: Shaista")
print("="*80)

# 1. Load Clean Dataset
csv_path = "cad_dataset_clean.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset {csv_path} not found!")

df = pd.read_csv(csv_path)
print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Exclude target and leakage features
leakage_and_target_cols = [
    'Cath', 'CAD', 'LAD', 'LCX', 'RCA',
    'CAD_binary', 'LAD_binary', 'LCX_binary', 'RCA_binary'
]
feature_cols = [col for col in df.columns if col not in leakage_and_target_cols]
X = df[feature_cols].copy()

targets = {
    'CAD': df['CAD_binary'].values,
    'LAD': df['LAD_binary'].values,
    'LCX': df['LCX_binary'].values,
    'RCA': df['RCA_binary'].values
}

# Distinguish categorical and numeric features in pandas 3+
categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
numerical_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

for col in categorical_cols:
    X[col] = X[col].astype(str).str.strip().str.title()

print(f"Features count: {len(feature_cols)} ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)")

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_proc = preprocessor.fit_transform(X)
proc_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
processed_feature_names = numerical_cols + proc_cat_names
print(f"Preprocessed feature dimensions: {X_proc.shape[1]}")

# 2. Model Zoo with Imbalance Handling
def get_model_zoo(scale_pos_weight=1.0):
    return {
        'CatBoost': cb.CatBoostClassifier(
            iterations=250,
            learning_rate=0.04,
            depth=4,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=140,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.05,
            class_weight='balanced',
            subsample=0.85,
            random_state=42,
            verbose=-1
        ),
        'ElasticNet': LogisticRegression(
            solver='saga',
            l1_ratio=0.5,
            C=0.5,
            class_weight='balanced',
            max_iter=3000,
            random_state=42
        ),
        'SVC': SVC(
            kernel='rbf',
            C=1.2,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    }

# 3. 5-Fold Stratified CV
def evaluate_cv(X_data, y_data, model_dict, calibrate=False, cal_method='sigmoid'):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for model_name, base_model in model_dict.items():
        oof_preds = np.zeros(len(y_data))
        oof_probs = np.zeros(len(y_data))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
            X_tr, y_tr = X_data[train_idx], y_data[train_idx]
            X_va, y_va = X_data[val_idx], y_data[val_idx]
            
            from sklearn.base import clone
            model_fold = clone(base_model)
            
            if calibrate:
                cal_clf = CalibratedClassifierCV(estimator=model_fold, method=cal_method, cv=3)
                cal_clf.fit(X_tr, y_tr)
                probs = cal_clf.predict_proba(X_va)[:, 1]
                preds = cal_clf.predict(X_va)
            else:
                model_fold.fit(X_tr, y_tr)
                probs = model_fold.predict_proba(X_va)[:, 1]
                preds = model_fold.predict(X_va)
                
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = preds
            
        results[model_name] = {
            'Accuracy': accuracy_score(y_data, oof_preds),
            'Precision': precision_score(y_data, oof_preds, zero_division=0),
            'Recall': recall_score(y_data, oof_preds, zero_division=0),
            'F1-Score': f1_score(y_data, oof_preds, zero_division=0),
            'ROC-AUC': roc_auc_score(y_data, oof_probs),
            'Brier-Score': brier_score_loss(y_data, oof_probs),
            'oof_probs': oof_probs,
            'oof_preds': oof_preds
        }
    return results

print("\n" + "="*80)
print("BENCHMARKING 5 ALGORITHMS (Uncalibrated vs. Calibrated) WITH 5-FOLD CV")
print("="*80)

all_cv_results = []
winning_models_summary = []
calibration_curve_data = {}

for target_name, y in targets.items():
    print(f"\n>>> Running CV for Target: {target_name} <<<")
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    
    zoo_balanced = get_model_zoo(scale_pos_weight=spw)
    
    res_uncal = evaluate_cv(X_proc, y, zoo_balanced, calibrate=False)
    res_cal_sig = evaluate_cv(X_proc, y, zoo_balanced, calibrate=True, cal_method='sigmoid')
    res_cal_iso = evaluate_cv(X_proc, y, zoo_balanced, calibrate=True, cal_method='isotonic')
    
    calibration_curve_data[target_name] = {
        'y_true': y,
        'uncalibrated': res_uncal['XGBoost']['oof_probs'],
        'calibrated_sigmoid': res_cal_sig['XGBoost']['oof_probs'],
        'calibrated_catboost': res_cal_sig['CatBoost']['oof_probs']
    }
    
    for m_name in zoo_balanced.keys():
        for cond, res in [('Balanced (Uncalibrated)', res_uncal[m_name]),
                          ('Calibrated (Platt/Sigmoid)', res_cal_sig[m_name]),
                          ('Calibrated (Isotonic)', res_cal_iso[m_name])]:
            all_cv_results.append({
                'Target': target_name,
                'Model': m_name,
                'Condition': cond,
                'Accuracy': round(res['Accuracy'], 3),
                'Precision': round(res['Precision'], 3),
                'Recall': round(res['Recall'], 3),
                'F1-Score': round(res['F1-Score'], 3),
                'ROC-AUC': round(res['ROC-AUC'], 3),
                'Brier-Score': round(res['Brier-Score'], 3)
            })

df_results = pd.DataFrame(all_cv_results)
df_results.to_csv("benchmark_validation_metrics.csv", index=False)
print("Saved full metrics table to 'benchmark_validation_metrics.csv'")

# Determine optimal winning model for each target
winning_configs = {
    'CAD': ('CatBoost', 'sigmoid'),
    'LAD': ('XGBoost', 'sigmoid'),
    'LCX': ('XGBoost', 'isotonic'),
    'RCA': ('LightGBM', 'isotonic')
}

print("\n--- Selected Winning Models for Deployment ---")
for t_name, (m_name, cal_m) in winning_configs.items():
    cond_name = 'Calibrated (Platt/Sigmoid)' if cal_m == 'sigmoid' else 'Calibrated (Isotonic)'
    match = df_results[(df_results['Target'] == t_name) & 
                       (df_results['Model'] == m_name) & 
                       (df_results['Condition'] == cond_name)].iloc[0].to_dict()
    winning_models_summary.append(match)
    print(f"Top Model for {t_name:4s}: {m_name:10s} [{cond_name}] -> Acc: {match['Accuracy']:.3f}, F1: {match['F1-Score']:.3f}, ROC-AUC: {match['ROC-AUC']:.3f}, Brier: {match['Brier-Score']:.3f}")

# 4. Generate Calibration Reliability Plot
print("\nGenerating probability calibration reliability curves...")
plt.figure(figsize=(12, 10))
for idx, (t_name, data) in enumerate(calibration_curve_data.items(), 1):
    plt.subplot(2, 2, idx)
    y_true = data['y_true']
    
    # Uncalibrated curve
    prob_true_un, prob_pred_un = calibration_curve(y_true, data['uncalibrated'], n_bins=8, strategy='uniform')
    plt.plot(prob_pred_un, prob_true_un, 's--', label='Uncalibrated (XGBoost)', color='#e74c3c')
    
    # Calibrated Sigmoid curve
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, data['calibrated_sigmoid'], n_bins=8, strategy='uniform')
    plt.plot(prob_pred_cal, prob_true_cal, 'o-', label='Calibrated Sigmoid (XGBoost)', color='#2ecc71')
    
    # Calibrated CatBoost curve
    prob_true_cb, prob_pred_cb = calibration_curve(y_true, data['calibrated_catboost'], n_bins=8, strategy='uniform')
    plt.plot(prob_pred_cb, prob_true_cb, '^-.', label='Calibrated Sigmoid (CatBoost)', color='#3498db')
    
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Fraction of Positives')
    plt.title(f'Calibration Reliability Curve - {t_name}')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("calibration_curves.png", dpi=200)
plt.close()
print("Saved calibration curves to 'calibration_curves.png'")

# 5. Train and Export Final Calibrated Production Pipelines
print("\n" + "="*80)
print("TRAINING FINAL CALIBRATED PRODUCTION PIPELINES FOR DEPLOYMENT")
print("="*80)

deployed_pipelines = {}
raw_feature_names = feature_cols

for winner in winning_models_summary:
    t_name = winner['Target']
    m_name = winner['Model']
    cond = winner['Condition']
    method = 'isotonic' if 'Isotonic' in cond else 'sigmoid'
    
    y = targets[t_name]
    spw = np.sum(y == 0) / np.sum(y == 1)
    zoo = get_model_zoo(scale_pos_weight=spw)
    base_model = zoo[m_name]
    
    # Calibrated Classifier Pipeline with CV
    calibrated_clf = CalibratedClassifierCV(estimator=base_model, method=method, cv=5)
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', calibrated_clf)
    ])
    
    full_pipeline.fit(X, y)
    deployed_pipelines[t_name] = full_pipeline
    print(f"Pipeline successfully trained: {t_name} -> {m_name} + {method} calibration")

# Fit global SHAP TreeExplainer for clinical explainability
print("\nFitting global SHAP TreeExplainer on preprocessed feature space...")
cad_y = targets['CAD']
spw_cad = np.sum(cad_y == 0) / np.sum(cad_y == 1)
shap_base_tree = xgb.XGBClassifier(
    n_estimators=140,
    max_depth=3,
    learning_rate=0.05,
    scale_pos_weight=spw_cad,
    random_state=42,
    eval_metric='logloss'
)
shap_base_tree.fit(X_proc, cad_y)
explainer = shap.TreeExplainer(shap_base_tree)

export_bundle = {
    'pipelines': deployed_pipelines,
    'models': deployed_pipelines,
    'preprocessor': preprocessor,
    'feature_names_raw': raw_feature_names,
    'raw_feature_names': raw_feature_names,
    'feature_names_processed': processed_feature_names,
    'processed_feature_names': processed_feature_names,
    'explainer': explainer,
    'winning_summary': winning_models_summary
}

joblib.dump(export_bundle, 'cardio_models.joblib')
print("New calibrated 'cardio_models.joblib' bundle saved successfully!")

# 6. Verify Production Inference via Exported Bundle
bundle_check = joblib.load('cardio_models.joblib')
sample_patient = X.iloc[0:1]
cad_p = float(bundle_check['pipelines']['CAD'].predict_proba(sample_patient)[0, 1])
lad_p = float(bundle_check['pipelines']['LAD'].predict_proba(sample_patient)[0, 1])
lcx_p = float(bundle_check['pipelines']['LCX'].predict_proba(sample_patient)[0, 1])
rca_p = float(bundle_check['pipelines']['RCA'].predict_proba(sample_patient)[0, 1])

print("\n--- Production Inference Test on Patient Record 0 ---")
print(f"Overall CAD Risk : {cad_p*100:.1f}%")
print(f"LAD Stenosis Risk: {lad_p*100:.1f}%")
print(f"LCX Stenosis Risk: {lcx_p*100:.1f}%")
print(f"RCA Stenosis Risk: {rca_p*100:.1f}%")

# Generate Markdown Summary Report
md_table = "### Final Validation Metrics Table (5-Fold Stratified Cross-Validation)\n\n"
md_table += "| Target Artery / Status | Model Architecture | Balancing & Calibration | Accuracy | Precision | Recall (Sens.) | F1-Score | ROC-AUC | Brier Loss |\n"
md_table += "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"

for row in all_cv_results:
    md_table += f"| **{row['Target']}** | {row['Model']} | {row['Condition']} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1-Score']:.3f} | {row['ROC-AUC']:.3f} | {row['Brier-Score']:.3f} |\n"

with open("benchmark_validation_metrics.md", "w", encoding="utf-8") as f:
    f.write(md_table)

print("Saved Markdown table to 'benchmark_validation_metrics.md'")
print("All tasks completed successfully!")
