import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

def evaluate_fusion_model():
    print("="*50)
    print(" COPD MULTIMODAL FUSION - EVALUATION REPORT ")
    print("="*50)
    
    # 1. Load the Meta-Learner
    model_path = os.path.join('models', 'fusion_logistic_regression.joblib')
    if not os.path.exists(model_path):
        print("[ERROR] Fusion model not found. Cannot evaluate.")
        return
        
    meta_model = joblib.load(model_path)
    
    # 2. Generate a Synthetic "Held-Out Test Set" (Simulating 5,000 new patients)
    print("\nGenerating 5,000 synthetic test patients with real-world biological variance...")
    np.random.seed(99)
    n_samples = 5000
    
    clinical_test = np.random.uniform(0, 1, n_samples)
    xray_test = np.random.uniform(0, 1, n_samples)
    audio_test = np.random.uniform(0, 1, n_samples)
    
    X_test = np.column_stack((clinical_test, xray_test, audio_test))
    
    # Add heavy Gaussian noise to simulate messy real-world clinical data
    # (e.g. patients with healthy X-Rays but severe Clinical COPD)
    biological_noise = np.random.normal(0, 0.35, n_samples)
    true_risk_score = 0.4 * clinical_test + 0.3 * xray_test + 0.3 * audio_test + biological_noise
    
    y_true = (true_risk_score > 0.5).astype(int)
    
    # 3. Run Inference on the Test Set
    print("Running Meta-Learner inference...")
    y_pred_probs = meta_model.predict_proba(X_test)[:, 1]
    y_pred_labels = (y_pred_probs > 0.5).astype(int)
    
    # 4. Calculate Metrics
    acc = accuracy_score(y_true, y_pred_labels)
    prec = precision_score(y_true, y_pred_labels)
    rec = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    
    # 5. Print Results
    print("\n--- STATISTICAL METRICS ---")
    print(f"Accuracy:    {acc*100:.2f}%  (Overall correctness)")
    print(f"Precision:   {prec*100:.2f}%  (When it predicts COPD, how often is it right?)")
    print(f"Recall:      {rec*100:.2f}%  (Out of all sick patients, how many did it catch?)")
    print(f"F1-Score:    {f1*100:.2f}%  (Balance of Precision and Recall)")
    print(f"ROC-AUC:     {roc_auc:.4f}   (Ability to distinguish healthy vs sick)")
    
    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred_labels, target_names=["Healthy", "COPD"]))
    print("="*50)

if __name__ == "__main__":
    evaluate_fusion_model()
