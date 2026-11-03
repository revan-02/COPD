# Multimodal COPD Research Pipeline — Dataset Setup

This document outlines the datasets required for the Clinical Spirometry and Respiratory Audio modalities.
**DO NOT INVENT DATASETS. DO NOT FABRICATE PATIENTS.**

## 1. Clinical Data (Spirometry & Demographics)
**Model:** XGBoost Classifier (`models/xgb_copd_model.joblib`)
- **Features Used:**
  - FEV1 (Forced Expiratory Volume in 1 second, mL)
  - FVC (Forced Vital Capacity, mL)
  - Age (years)
  - Height (cm)
  - Gender (Male/Female)
  - Smoking Status (Never/Former/Current)
  - Race Reference Equation (Caucasian, African American, Mexican American, Other)
- **Clinical Basis:** GOLD Guidelines (FEV1/FVC < 0.70 post/pre-bronchodilator threshold for airflow limitation).

## 2. Respiratory Audio (CRNN)
**Required Public Dataset:** ICBHI 2017 Respiratory Sound Database
- **Model:** BiLSTM + CNN (`models/audio_crnn.keras`)
- **Usage:** Identifies adventitious lung sounds (crackles, wheezes).
- **Labels Required:** Normal breathing, Crackle detected, Wheeze detected, Both (Crackle + Wheeze).
- **Expected Structure:**
  - `[PatientID]_[...].wav`
  - `[PatientID]_[...].txt`
- **Preprocessing:** Resample to 16kHz, extract bounding box segments, convert to 128-bin Mel-spectrogram, pad/trim to 259 frames.
- **Split:** `GroupShuffleSplit` on `Patient ID` to prevent patient leakage across train/test splits.
- **Training:** Run `python training/train_audio.py`.
- **Connecting:** Save output as `models/audio_crnn.keras`.

## 3. Late Fusion (Logistic Regression)
- **Model:** Logistic Regression Meta-Learner (`models/fusion_logistic_regression.joblib`)
- **Usage:** Fuses Clinical Spirometry Probability + Respiratory Audio Abnormal Probability (2-feature late fusion matrix `[clinical_prob, audio_prob]`).
- **Training Data:** Uses Out-of-Fold (OOF) validation probabilities or simulated multi-modal combinations.
- **Training:** Run `python training/train_fusion_no_xray.py`.
- **Connecting:** Save output as `models/fusion_logistic_regression.joblib`.

## 4. Live Audio & Hardware Integration
- Real-time respiratory acoustic recording via WebSockets (`/live` or direct microphone stream).
- Rolling audio buffer (5.0s window, 16kHz sampling rate) continuously passed to `inference/audio_inference.py`.

## 5. Demo Mode
If the models in `models/` do not exist, the UI will correctly display **MODEL NOT LOADED**. It will not generate fake probabilities.
