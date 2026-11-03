import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Features: ['FEV1', 'FVC', 'FEV1PRED', 'FVCPRED', 'AGE', 'GENDER_ENC', 'SMOKE_former', 'SMOKE_never']
# Target: COPD (0 or 1)

np.random.seed(42)
n_samples = 1000

X = pd.DataFrame({
    'FEV1': np.random.uniform(1000, 4000, n_samples),
    'FVC': np.random.uniform(1500, 5000, n_samples),
    'FEV1PRED': np.random.uniform(40, 120, n_samples),
    'FVCPRED': np.random.uniform(40, 120, n_samples),
    'AGE': np.random.uniform(40, 80, n_samples),
    'GENDER_ENC': np.random.randint(0, 2, n_samples),
    'SMOKE_former': np.random.randint(0, 2, n_samples),
    'SMOKE_never': np.random.randint(0, 2, n_samples)
})

X['FEV1'] = np.minimum(X['FEV1'], X['FVC'] - 100)

ratio = X['FEV1'] / X['FVC']
prob_copd = 1.0 - (ratio - 0.3) / 0.5
prob_copd = np.clip(prob_copd, 0.05, 0.95)
prob_copd += (X['AGE'] - 50) * 0.01
prob_copd += (1 - X['SMOKE_never']) * 0.15
prob_copd = np.clip(prob_copd, 0.01, 0.99)

y = np.random.binomial(1, prob_copd)

model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

joblib.dump(model, 'xgb_copd_model.joblib')
print("Dummy model trained and saved successfully.")
