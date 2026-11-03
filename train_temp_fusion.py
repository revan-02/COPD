import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Generate synthetic 3-feature training data
np.random.seed(42)
n_samples = 1500

clinical = np.random.uniform(0, 1, n_samples)
xray = np.random.uniform(0, 1, n_samples)
audio = np.random.uniform(0, 1, n_samples)
X = np.column_stack((clinical, xray, audio))

# Add Gaussian noise to simulate biological variance and imperfect labels
noise = np.random.normal(0, 0.25, n_samples)
continuous_risk = 0.4 * clinical + 0.3 * xray + 0.3 * audio + noise

# Ground truth is based on the noisy risk threshold
y = (continuous_risk > 0.5).astype(int)

# Train Meta-Learner
meta_model = LogisticRegression()
meta_model.fit(X, y)

# Save the 3-feature model
joblib.dump(meta_model, 'models/fusion_logistic_regression.joblib')
print("Successfully trained full 3-feature Fusion model!")
