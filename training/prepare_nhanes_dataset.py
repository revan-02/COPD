import os
import pandas as pd
import numpy as np

# Ensure data output directory exists
os.makedirs('data', exist_ok=True)

# 1. Load raw NHANES tables
spx_path = 'data/nhanes_raw/SPX_E.XPT'
demo_path = 'data/nhanes_raw/DEMO_E.xpt'
bmx_path = 'data/nhanes_raw/BMX_E.xpt'
smq_path = 'data/nhanes_raw/SMQ_E.xpt'

print("Loading raw CDC NHANES SAS transport files...")
spx = pd.read_sas(spx_path, format='xport')[['SEQN', 'SPXNFEV1', 'SPXNFVC']]
demo = pd.read_sas(demo_path, format='xport')[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1']]
bmx = pd.read_sas(bmx_path, format='xport')[['SEQN', 'BMXHT']]
smq = pd.read_sas(smq_path, format='xport')[['SEQN', 'SMQ020', 'SMQ040']]

# Merge tables on unique participant identifier (SEQN)
df = spx.merge(demo, on='SEQN').merge(bmx, on='SEQN').merge(smq, on='SEQN', how='left')
print(f"Total merged records: {len(df)}")

# Filter out missing records in critical measurements
df = df.dropna(subset=['SPXNFEV1', 'SPXNFVC', 'RIDAGEYR', 'RIAGENDR', 'BMXHT', 'RIDRETH1'])

# Focus on adult cohort (Age >= 20) where spirometric obstruction is diagnostically evaluated
df = df[df['RIDAGEYR'] >= 20].copy()
print(f"Valid adult spirometry records (Age >= 20): {len(df)}")

# 2. Map Race codes to project naming:
# RIDRETH1: 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White, 4=Non-Hispanic Black, 5=Other Race
def map_race(r):
    if r == 4:
        return 'african_american'
    elif r == 1:
        return 'mexican_american'
    elif r == 3:
        return 'caucasian'
    else:
        return 'other'

df['race_category'] = df['RIDRETH1'].apply(map_race)
df['gender_category'] = df['RIAGENDR'].apply(lambda g: 'male' if g == 1 else 'female')

# 3. Compute NHANES III predicted spirometry reference values
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

pred_fev1_list = []
pred_fvc_list = []
for idx, row in df.iterrows():
    p_fev1, p_fvc = calculate_predicted_spirometry(row['RIDAGEYR'], row['BMXHT'], row['gender_category'], row['race_category'])
    pred_fev1_list.append(p_fev1)
    pred_fvc_list.append(p_fvc)

df['FEV1_REF'] = pred_fev1_list
df['FVC_REF'] = pred_fvc_list

# Compute Percent Predicted
df['FEV1PRED'] = (df['SPXNFEV1'] / df['FEV1_REF']) * 100.0
df['FVCPRED'] = (df['SPXNFVC'] / df['FVC_REF']) * 100.0

# Rename to exact model feature conventions
df['FEV1'] = df['SPXNFEV1']
df['FVC'] = df['SPXNFVC']
df['AGE'] = df['RIDAGEYR']
df['GENDER_ENC'] = df['RIAGENDR'].astype(int) # 1 = Male, 2 = Female

# Smoking status encoding:
# SMQ020: 1 = Smoked 100+ cigarettes; 2 = No
# SMQ040: 1 = Every day, 2 = Some days, 3 = Not at all
df['SMOKE_never'] = (df['SMQ020'] == 2).astype(int)
df['SMOKE_former'] = ((df['SMQ020'] == 1) & (df['SMQ040'] == 3)).astype(int)

# 4. Define Clinical COPD Target based on GOLD standard (FEV1/FVC < 0.70)
df['FEV1_FVC_RATIO'] = df['FEV1'] / df['FVC']
df['COPD'] = (df['FEV1_FVC_RATIO'] < 0.70).astype(int)

# GOLD Severity Staging for COPD positive cases:
def assign_gold_stage(row):
    if row['COPD'] == 0:
        return 'Normal'
    fev1p = row['FEV1PRED']
    if fev1p >= 80.0:
        return 'GOLD 1 (Mild)'
    elif fev1p >= 50.0:
        return 'GOLD 2 (Moderate)'
    elif fev1p >= 30.0:
        return 'GOLD 3 (Severe)'
    else:
        return 'GOLD 4 (Very Severe)'

df['GOLD_STAGE'] = df.apply(assign_gold_stage, axis=1)

# Select final features
clean_cols = [
    'SEQN', 'FEV1', 'FVC', 'FEV1PRED', 'FVCPRED', 'AGE', 'GENDER_ENC',
    'SMOKE_former', 'SMOKE_never', 'FEV1_FVC_RATIO', 'COPD', 'GOLD_STAGE'
]
final_df = df[clean_cols].copy()

output_csv = 'data/nhanes_copd_clinical_dataset.csv'
final_df.to_csv(output_csv, index=False)
print(f"\nSuccessfully generated {output_csv} with {len(final_df)} patient records!")
print("\nTarget Class Distribution:")
print(final_df['COPD'].value_counts())
print("\nGOLD Severity Stage Breakdown:")
print(final_df['GOLD_STAGE'].value_counts())
