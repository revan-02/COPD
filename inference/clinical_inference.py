import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_copd_model.joblib')

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
except Exception:
    model = None
    MODEL_LOADED = False

def calculate_predicted_spirometry(age, height_cm, gender, race):
    age = float(age)
    h2 = float(height_cm) ** 2
    a = age
    a2 = age ** 2
    is_male = (gender == 'male')

    if is_male:
        if race == 'african_american':
            fev1_L = 0.3411 - 0.01283 * a - 0.000108 * a2 + 0.00010816 * h2
            fvc_L  = -0.4971 - 0.00495 * a - 0.000183 * a2 + 0.00014722 * h2
        elif race == 'mexican_american':
            fev1_L = 0.6306 - 0.01657 * a - 0.000129 * a2 + 0.00013801 * h2
            fvc_L  = 0.2376 - 0.00891 * a - 0.000182 * a2 + 0.00016643 * h2
        elif race == 'other':
            fev1_L = (0.5536 - 0.01303 * a - 0.000172 * a2 + 0.00014098 * h2) * 0.88
            fvc_L  = (-0.1933 - 0.00694 * a - 0.000216 * a2 + 0.00018642 * h2) * 0.88
        else: # caucasian
            fev1_L = 0.5536 - 0.01303 * a - 0.000172 * a2 + 0.00014098 * h2
            fvc_L  = -0.1933 - 0.00694 * a - 0.000216 * a2 + 0.00018642 * h2
    else:
        if race == 'african_american':
            fev1_L = 0.3433 - 0.01283 * a - 0.000085 * a2 + 0.00009623 * h2
            fvc_L  = -0.3541 - 0.00344 * a - 0.000170 * a2 + 0.00012270 * h2
        elif race == 'mexican_american':
            fev1_L = 0.4529 - 0.01178 * a - 0.000113 * a2 + 0.00011249 * h2
            fvc_L  = 0.1210 - 0.00307 * a - 0.000183 * a2 + 0.00013492 * h2
        elif race == 'other':
            fev1_L = (0.4333 - 0.00361 * a - 0.000194 * a2 + 0.00011496 * h2) * 0.88
            fvc_L  = (0.1210 + 0.00307 * a - 0.000237 * a2 + 0.00014497 * h2) * 0.88
        else: # caucasian
            fev1_L = 0.4333 - 0.00361 * a - 0.000194 * a2 + 0.00011496 * h2
            fvc_L  = 0.1210 + 0.00307 * a - 0.000237 * a2 + 0.00014497 * h2

    return max(fev1_L * 1000.0, 1.0), max(fvc_L * 1000.0, 1.0)

def predict_clinical(form_data):
    if not MODEL_LOADED:
        return None
        
    fev1 = float(form_data.get('fev1', 0))
    fvc = float(form_data.get('fvc', 0))
    age = float(form_data.get('age', 0))
    height = float(form_data.get('height', 0))
    gender = form_data.get('gender', 'male')
    race = form_data.get('race', 'caucasian')
    smoking_status = form_data.get('smoking_status', 'never')

    # Fail-safe auto-detection: if user entered in Liters (e.g. 1.0 - 10.0), convert to mL
    if 0 < fev1 <= 15.0:
        fev1 = fev1 * 1000.0
    if 0 < fvc <= 15.0:
        fvc = fvc * 1000.0

    fev1_ref, fvc_ref = calculate_predicted_spirometry(age, height, gender, race)
    fev1pred_pct = (fev1 / fev1_ref) * 100.0
    fvcpred_pct = (fvc / fvc_ref) * 100.0

    gender_enc = 1 if gender == 'male' else 2
    smoke_former = 1 if smoking_status == 'former' else 0
    smoke_never = 1 if smoking_status == 'never' else 0

    input_df = pd.DataFrame([{
        'FEV1': fev1, 'FVC': fvc, 'FEV1PRED': fev1pred_pct, 'FVCPRED': fvcpred_pct,
        'AGE': age, 'GENDER_ENC': gender_enc, 'SMOKE_former': smoke_former, 'SMOKE_never': smoke_never
    }])

    prob = float(model.predict_proba(input_df)[0][1])
    return prob
