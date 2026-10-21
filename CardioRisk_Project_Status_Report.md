# Multimodal AI Hackathon 2026 — Track A
# Cardiovascular Risk Visualization & Prediction
## Project Status, Milestone Audit & Team Work Distribution Report

**Date:** October 20, 2026  
**Project:** CardioRisk 3D Visualization & Prediction  
**Team Members:** Tooba (Lead), Shaista, Anamta, Anushka  

---

## 1. Executive Summary & Challenge Overview
Cardiovascular disease (CAD) is the leading cause of death worldwide. Current statistical risk scores produce abstract percentages that lack intuitive spatial context for clinicians and patients. 

**Our Objective:** Build an interactive 3D visualization and machine learning system that predicts:
1. Overall CAD status.
2. Vessel-specific stenosis ($ge 50%$) for the three critical coronary arteries:
   - **LAD** (Left Anterior Descending)
   - **LCX** (Left Circumflex)
   - **RCA** (Right Coronary Artery)
3. Maps risk probabilities onto an anatomical 3D heart model in real time.
4. Generates patient-level clinical explainability using SHAP (SHapley Additive exPlanations).

---

## 2. Completed Milestones (What Has Been Done Till Now)

| Area | Status | Deliverables Achieved |
| :--- | :---: | :--- |
| **Data Cleaning & Pipeline** | ✅ Done | UCI Z-Alizadeh Sani dataset (303 records, ~55 features) preprocessed, categorical encodings established, target leakage strictly prevented by isolating LAD, LCX, RCA, and Cath. |
| **Model & Explainer Bundle** | ✅ Done | Trained baseline models for all 4 targets bundled with SHAP TreeExplainer into `cardio_models.joblib`. |
| **FastAPI Backend Server** | ✅ Done | `server.py` created with REST endpoints (`/` health check, `/predict` inference) and CORS enabled for frontend communication. |
| **Interactive Frontend** | ✅ Done | React 19 + Vite dashboard featuring dynamic vitals input form, demo patient presets, and clinical explainability cards. |
| **3D Anatomical Foundation** | ✅ Done | Three.js + React Three Fiber heart model created with orbit controls, interactive camera, and dynamic vessel color coding (Normal: <30% Green, Moderate: 30–60% Amber, Stenotic: >60% Red). |
| **Bug Fixes & Stabilization**| ✅ Done | Resolved Windows directory ampersand command-split bug, installed missing frontend packages, added React 19 Drei Suspense boundaries, and integrated Tailwind CSS styling. |

---

## 3. Team Work Distribution & Ownership
*Every member owns a core domain tied to the hackathon evaluation rubric (Total 100%).*

### 🔬 Shaista — AI/ML Engineer
**Evaluation Rubric:** *Predictive Performance (30%)*
- **Role:** Model Evaluation, Tuning & Benchmarking
- **Specific Tasks:**
  1. Benchmark multiple algorithms: **CatBoost**, **XGBoost**, **LightGBM**, **ElasticNet/Logistic Regression**, and **Support Vector Classifiers (SVC)**.
  2. Implement probability calibration (`CalibratedClassifierCV`) so predicted percentages reflect genuine clinical risk.
  3. Handle vessel class imbalances for LCX and RCA using `class_weight='balanced'` or `scale_pos_weight`.
  4. Generate the final validation metrics table (Accuracy, Precision, Recall, F1-Score, ROC-AUC) comparing all models.
  5. Provide the winning model pipeline for deployment.

### 🎮 Anamta — Core CS & Graphics Engineer
**Evaluation Rubric:** *3D Visualization (25%) + Project Documentation (10%)*
- **Role:** 3D Anatomical Heart Model & Comprehensive Project Report
- **Specific Tasks:**
  1. Source and integrate a realistic open-source 3D heart mesh (`.glb` / `.gltf` format from Sketchfab, Three.js, or NIH 3D Print Exchange).
  2. Implement interactive vessel raycasting (mouse click/hover) so clinicians can click the LAD, LCX, or RCA to inspect local blood supply regions and stenosis risk.
  3. Add visual post-processing (stenosis glow effects, subtle pulse animations).
  4. **Lead Author for the Project Report:** Draft the mandatory 6-page comprehensive project documentation covering dataset preprocessing, model architectures, 3D pipeline setup, and clinical utility.

### ⚙️ Anushka — AI/ML & Systems Engineer
**Evaluation Rubric:** *Clinical Interpretability (20%) + System Integration (15%)*
- **Role:** Model-Code Integration & Explainability Pipeline
- **Specific Tasks:**
  1. Integrate the top-performing models from Shaista into `server.py` and test API response latency.
  2. Enhance the SHAP explanation engine to return both global feature importances and granular patient-level factor impacts (e.g., how specific lab values altered the risk score).
  3. Expand `demo_presets.json` with 4–5 realistic patient case studies (Low Risk, Single-vessel LAD, Multi-vessel High Risk).
  4. Ensure end-to-end data validation between frontend form inputs and backend preprocessing tensors.

### 👑 Tooba — Team Lead & Product Architect
**Evaluation Rubric:** *System Integration, UI/UX & Final Presentation*
- **Role:** End-to-End Orchestration, Frontend UI/UX, Video & Pitch Delivery
- **Specific Tasks:**
  1. Coordinate team deliverables and integrate all module updates into the central codebase.
  2. Polish frontend dashboard UX, responsive layout, and visual clinical safety disclaimers.
  3. Direct, record, and edit the **3–10 minute Demonstration Video** demonstrating the clinical workflow.
  4. Coordinate final pitch deck (PPT) preparation and present the hackathon solution.

---

## 4. Pending Tasks & Sprint Roadmap (What's Left To Be Done)

- [ ] **Sprint 1 (Models & 3D Assets):**
  - [x] Shaista completed the 5-algorithm benchmark comparison table (CatBoost, XGBoost, LightGBM, ElasticNet, SVC), implemented CalibratedClassifierCV, handled LCX/RCA vessel imbalances, and bundled calibrated pipelines into cardio_models.joblib.
  - Anamta selects and tests the `.gltf` heart mesh.
- [ ] **Sprint 2 (Integration & SHAP Refinement):**
  - Anushka updates `server.py` with calibrated models and detailed SHAP outputs.
  - Anamta wires vessel click/hover events in React Three Fiber.
- [ ] **Sprint 3 (Documentation & Demo):**
  - Anamta compiles the 6-page project report.
  - Team finalizes presentation slide deck (PPT).
  - Tooba records the 3-10 minute demonstration video.

---

## 5. Hackathon Evaluation Rubric Alignment
- **Predictive Performance (30%):** Owned by Shaista (CatBoost, XGBoost, ROC-AUC, 5-Fold Stratified CV).
- **3D Visualization (25%):** Owned by Anamta (Realistic mesh, raycasting, vessel color mapping).
- **Clinical Interpretability (20%):** Owned by Anushka (SHAP local/global attributions, factor impact).
- **System Integration (15%):** Owned by Tooba & Anushka (FastAPI + React 19 real-time inference).
- **Technical Implementation & Documentation (10%):** Owned by Anamta & Tooba (Code modularity, reproducibility, 6-page paper).
