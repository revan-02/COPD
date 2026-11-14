import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# 1. Load Real NHANES COPD Dataset
dataset_path = 'data/nhanes_copd_clinical_dataset.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"{dataset_path} not found. Run prepare_nhanes_dataset.py first.")

print(f"Loading real CDC NHANES dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)
print(f"Total Cohort Size: {len(df)} patients")

# Feature matrix matching the project's clinical inference pipeline
features = ['FEV1', 'FVC', 'FEV1PRED', 'FVCPRED', 'AGE', 'GENDER_ENC', 'SMOKE_former', 'SMOKE_never']
X = df[features]
y = df['COPD']

print(f"\nFeature set ({len(features)} variables): {features}")
print(f"Class distribution: Healthy/Non-COPD = {(y==0).sum()}, COPD = {(y==1).sum()} (Prevalence: {y.mean()*100:.1f}%)")

# 2. Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# 3. Model Training: Tuned XGBoost Classifier
print("\n" + "="*50)
print("TRAINING TUNED XGBOOST ON REAL NHANES DATA")
print("="*50)

# Compute scale_pos_weight to address class imbalance (~1:5.7 ratio)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=1.0,  # Standard calibration for balanced probabilities
    eval_metric='logloss',
    random_state=42
)

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

model.fit(X_train, y_train)

# 4. Model Evaluation on Held-Out Test Set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n--- HELD-OUT TEST EVALUATION METRICS ---")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print(f"ROC-AUC:   {auc:.4f}")

print("\n--- CONFUSION MATRIX ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Healthy / Non-COPD', 'COPD']))

# 5. Save Model to models/xgb_copd_model.joblib
output_model_path = 'models/xgb_copd_model.joblib'
joblib.dump(model, output_model_path)
print(f"\nSuccessfully saved trained real NHANES model to {output_model_path}!")

# 6. TARGET LEAKAGE AUDIT EXPERIMENT:
# Train a non-spirometric model using only demographics and smoking history
# to evaluate prediction capability when direct FEV1/FVC measurements are absent.
print("\n" + "="*50)
print("LEAKAGE-FREE EXPERIMENT (DEMOGRAPHICS & SMOKING ONLY)")
print("="*50)
leak_features = ['AGE', 'GENDER_ENC', 'SMOKE_former', 'SMOKE_never']
X_train_leak = X_train[leak_features]
X_test_leak = X_test[leak_features]

leak_model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)
leak_model.fit(X_train_leak, y_train)
y_pred_leak = leak_model.predict(X_test_leak)
y_prob_leak = leak_model.predict_proba(X_test_leak)[:, 1]

print(f"Leakage-Free Test Accuracy:  {accuracy_score(y_test, y_pred_leak)*100:.2f}%")
print(f"Leakage-Free Test Recall:    {recall_score(y_test, y_pred_leak)*100:.2f}%")
print(f"Leakage-Free Test ROC-AUC:   {roc_auc_score(y_test, y_prob_leak):.4f}")
print("==================================================")
