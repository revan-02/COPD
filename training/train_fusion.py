import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

print("Generating synthetic validation probabilities for Late Fusion...")
# We generate 1000 synthetic patient records
# Since we don't have true paired datasets, this is a proxy to learn the meta-weights.
# Features: [Clinical_Prob, Xray_Prob, Audio_Normal, Audio_Crackle, Audio_Wheeze, Audio_Both]

np.random.seed(42)
num_samples = 1500

# True label
y = np.random.randint(0, 2, num_samples)

# Simulate model probabilities based on true label (adding noise)
clinical_prob = np.clip(y * 0.7 + np.random.normal(0.1, 0.2, num_samples), 0, 1)
xray_prob = np.clip(y * 0.6 + np.random.normal(0.2, 0.3, num_samples), 0, 1)

# Audio probabilities (softmax distribution)
audio_probs = np.zeros((num_samples, 4))
for i in range(num_samples):
    if y[i] == 1:
        # COPD patient: higher chance of wheeze/both
        p = np.random.dirichlet([1, 2, 4, 3])
    else:
        # Normal patient: higher chance of normal
        p = np.random.dirichlet([5, 1, 1, 1])
    audio_probs[i] = p

X = np.column_stack([clinical_prob, xray_prob, audio_probs])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Training Fusion Meta-Learners ---")

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3),
    "XGBoost Stacker": XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500)
}

best_acc = 0
best_model_name = ""
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)
    print(f"{name:22s} -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

print(f"\nBest Fusion Model Selected: {best_model_name} (Accuracy: {best_acc:.4f})")
joblib.dump(best_model, 'fusion_meta_model.joblib')
print("Saved best model to fusion_meta_model.joblib")
