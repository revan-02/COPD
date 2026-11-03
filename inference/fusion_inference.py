import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fusion_logistic_regression.joblib')

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception:
    model = None
    MODEL_LOADED = False

def predict_fusion(clinical_prob, audio_prob):
    if not MODEL_LOADED or clinical_prob is None or audio_prob is None:
        return None
    if isinstance(audio_prob, str) or isinstance(clinical_prob, str):
        return None
        
    try:
        # 2-feature fusion meta-learner (Clinical + Audio)
        features = np.array([[clinical_prob, audio_prob]])
        prob = float(model.predict_proba(features)[0][1])
        return prob
    except Exception as e:
        print(f"Fusion inference error: {e}")
        return None
