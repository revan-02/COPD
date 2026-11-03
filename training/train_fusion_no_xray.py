import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

print("Generating synthetic validation probabilities for 2-feature Late Fusion...")
np.random.seed(42)
num_samples = 1500

y = np.random.randint(0, 2, num_samples)

clinical_prob = np.clip(y * 0.7 + np.random.normal(0.1, 0.2, num_samples), 0, 1)

# Audio probabilities (just taking the abnormal prob, similar to inference)
audio_prob = np.clip(y * 0.65 + np.random.normal(0.15, 0.25, num_samples), 0, 1)

X = np.column_stack([clinical_prob, audio_prob])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

acc = model.score(X_val, y_val)
print(f"2-Feature Fusion Model Accuracy: {acc:.4f}")

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fusion_logistic_regression.joblib')
joblib.dump(model, model_path)
print(f"Saved 2-feature model to {model_path}")
