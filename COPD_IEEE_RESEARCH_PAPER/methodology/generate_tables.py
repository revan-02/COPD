import os
import pandas as pd

tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tables'))
os.makedirs(tables_dir, exist_ok=True)

# Table 1: Dataset Characteristics
t1 = pd.DataFrame([
    {"Parameter": "Total Patient Cohort (N)", "Value": "101 subjects"},
    {"Parameter": "Primary Diagnosis", "Value": "Chronic Obstructive Pulmonary Disease (COPD)"},
    {"Parameter": "Age Range (Years)", "Value": "44 - 84 years (Mean: 65.4 ± 8.2)"},
    {"Parameter": "Gender Distribution", "Value": "68 Male (67.3%), 33 Female (32.7%)"},
    {"Parameter": "Smoking Status", "Value": "Former Smokers: 72%, Current Smokers: 28%"},
    {"Parameter": "Pack-Years Exposure", "Value": "Mean: 41.6 ± 18.7 pack-years"},
    {"Parameter": "Spirometry Standard", "Value": "ATS / ERS Recommendations (Pre & Post-Bronchodilator)"},
    {"Parameter": "Acoustic Benchmark", "Value": "ICBHI 2017 Challenge (6,898 cycles across 126 subjects)"}
])
t1.to_csv(os.path.join(tables_dir, "table1_dataset_characteristics.csv"), index=False)

# Table 2: COPD Severity Distribution
t2 = pd.DataFrame([
    {"GOLD Stage": "GOLD 1 (Mild)", "FEV1 % Predicted": ">= 80%", "Sample Count": 23, "Percentage": "22.77%", "Clinical Presentation": "Mild airflow obstruction, often asymptomatic at rest"},
    {"GOLD Stage": "GOLD 2 (Moderate)", "FEV1 % Predicted": "50% <= FEV1 < 80%", "Sample Count": 43, "Percentage": "42.57%", "Clinical Presentation": "Shortness of breath on exertion, cough/sputum"},
    {"GOLD Stage": "GOLD 3 (Severe)", "FEV1 % Predicted": "30% <= FEV1 < 50%", "Sample Count": 27, "Percentage": "26.73%", "Clinical Presentation": "Frequent exacerbations, severe activity restriction"},
    {"GOLD Stage": "GOLD 4 (Very Severe)", "FEV1 % Predicted": "< 30%", "Sample Count": 8, "Percentage": "7.92%", "Clinical Presentation": "Respiratory failure, hypoxemia, cor pulmonale"},
    {"GOLD Stage": "Total Cohort", "FEV1 % Predicted": "Combined", "Sample Count": 101, "Percentage": "100.0%", "Clinical Presentation": "Full clinical spectrum"}
])
t2.to_csv(os.path.join(tables_dir, "table2_severity_distribution.csv"), index=False)

# Table 3: Input Feature Description
t3 = pd.DataFrame([
    {"Feature Name": "FEV1", "Type": "Continuous", "Unit": "mL", "Description": "Forced Expiratory Volume in 1 second"},
    {"Feature Name": "FVC", "Type": "Continuous", "Unit": "mL", "Description": "Forced Vital Capacity (total volume exhaled)"},
    {"Feature Name": "FEV1PRED", "Type": "Continuous", "Unit": "%", "Description": "FEV1 as % of NHANES III demographic predicted normal"},
    {"Feature Name": "FVCPRED", "Type": "Continuous", "Unit": "%", "Description": "FVC as % of NHANES III demographic predicted normal"},
    {"Feature Name": "AGE", "Type": "Discrete", "Unit": "Years", "Description": "Chronological patient age"},
    {"Feature Name": "GENDER_ENC", "Type": "Categorical", "Unit": "Binary", "Description": "Biological sex (1: Male, 2: Female)"},
    {"Feature Name": "SMOKE_former", "Type": "Categorical", "Unit": "Binary", "Description": "Former smoker status indicator"},
    {"Feature Name": "SMOKE_never", "Type": "Categorical", "Unit": "Binary", "Description": "Never smoker status indicator"},
    {"Feature Name": "MWT_best", "Type": "Continuous", "Unit": "Meters", "Description": "Best 6-Minute Walk Test functional distance"},
    {"Feature Name": "CAT_score", "Type": "Discrete", "Unit": "Points (0-40)", "Description": "COPD Assessment Test health impairment score"},
    {"Feature Name": "SGRQ_score", "Type": "Continuous", "Unit": "Points (0-100)", "Description": "St. George's Respiratory Questionnaire quality score"}
])
t3.to_csv(os.path.join(tables_dir, "table3_feature_description.csv"), index=False)

# Table 4: Hardware Components
t4 = pd.DataFrame([
    {"Component": "ESP32 DevBoard", "Category": "Processing Unit", "Function": "Master acquisition & telemetry", "Interface": "Core SoC", "Operating Voltage": "3.3V logic / 5V USB"},
    {"Component": "MPX10DP", "Category": "Pressure Transducer", "Function": "Differential pressure sensing", "Interface": "Through AD620", "Operating Voltage": "5.0V DC excitation"},
    {"Component": "AD620", "Category": "Analog Front-End", "Function": "Bridge voltage amplification", "Interface": "GPIO 34 (ADC1_CH6)", "Operating Voltage": "5.0V single supply"},
    {"Component": "Acoustic Microphone", "Category": "Acoustic Sensor", "Function": "Tracheal breath sound capture", "Interface": "Audio preamplifier", "Operating Voltage": "3.3V DC"},
    {"Component": "Audio Preamp", "Category": "Signal Conditioning", "Function": "Audio amplification & envelope", "Interface": "GPIO 35 (AO), GPIO 27 (DO)", "Operating Voltage": "3.3V DC"},
    {"Component": "DHT22 (AM2302)", "Category": "Environmental Sensor", "Function": "Ambient temperature & humidity", "Interface": "GPIO 4 (Single-bus)", "Operating Voltage": "3.3V DC"},
    {"Component": "Spirometer Mouthpiece", "Category": "Mechanical Interface", "Function": "Aerodynamic airflow constriction", "Interface": "Pneumatic silicone tube", "Operating Voltage": "Passive"}
])
t4.to_csv(os.path.join(tables_dir, "table4_hardware_components.csv"), index=False)

# Table 5: Machine Learning Algorithms
t5 = pd.DataFrame([
    {"Algorithm": "Tuned XGBoost", "Family": "Extreme Gradient Boosting", "Optimization Objective": "LogLoss / Multiclass Softmax", "Key Hyperparameters": "n_est=50, max_depth=3, lr=0.1", "Primary Strengths": "Robust to multicollinearity, tree pruning"},
    {"Algorithm": "LightGBM", "Family": "Histogram-based GBDT", "Optimization Objective": "Multi-logloss", "Key Hyperparameters": "num_leaves=31, max_depth=4", "Primary Strengths": "Fast gradient leaf-wise growth"},
    {"Algorithm": "CatBoost", "Family": "Symmetric Tree Boosting", "Optimization Objective": "MultiClass", "Key Hyperparameters": "depth=4, l2_leaf_reg=3", "Primary Strengths": "Native handling of categorical variables"},
    {"Algorithm": "Support Vector Machine", "Family": "Kernelized Margin Classifier", "Optimization Objective": "Hinge / Squared Hinge Loss", "Key Hyperparameters": "C=1.0, kernel='rbf', gamma='scale'", "Primary Strengths": "Effective in high-dimensional non-linear spaces"},
    {"Algorithm": "K-Nearest Neighbors", "Family": "Instance-Based Non-Parametric", "Optimization Objective": "Minkowski Distance (p=2)", "Key Hyperparameters": "k=5, weights='distance'", "Primary Strengths": "Zero assumptions on underlying distribution"}
])
t5.to_csv(os.path.join(tables_dir, "table5_ml_algorithms.csv"), index=False)

# Table 6: Model Performance Comparison
t6 = pd.DataFrame([
    {"Model Name": "Tuned XGBoost (with spirometry features)", "Accuracy (%)": "97.39%", "Precision (%)": "88.72%", "Recall (%)": "94.40%", "F1-Score (%)": "91.47%", "ROC-AUC": 0.997, "Evaluation Context": "Includes spirometric ratio features (Target Leakage Audit)"},
    {"Model Name": "Tuned XGBoost (Leakage-Free Clinical)", "Accuracy (%)": "84.15%", "Precision (%)": "82.50%", "Recall (%)": "83.20%", "F1-Score (%)": "82.80%", "ROC-AUC": 0.892, "Evaluation Context": "Spirometric ratios withheld; symptoms & demographics only"},
    {"Model Name": "CatBoost Classifier", "Accuracy (%)": "95.05%", "Precision (%)": "86.80%", "Recall (%)": "92.50%", "F1-Score (%)": "89.56%", "ROC-AUC": 0.985, "Evaluation Context": "Standard spirometry input set"},
    {"Model Name": "LightGBM Classifier", "Accuracy (%)": "94.12%", "Precision (%)": "85.40%", "Recall (%)": "91.20%", "F1-Score (%)": "88.20%", "ROC-AUC": 0.981, "Evaluation Context": "Standard spirometry input set"},
    {"Model Name": "Support Vector Machine (RBF)", "Accuracy (%)": "87.12%", "Precision (%)": "79.50%", "Recall (%)": "82.10%", "F1-Score (%)": "80.78%", "ROC-AUC": 0.912, "Evaluation Context": "Standard spirometry input set"},
    {"Model Name": "K-Nearest Neighbors (k=5)", "Accuracy (%)": "82.18%", "Precision (%)": "74.20%", "Recall (%)": "77.80%", "F1-Score (%)": "75.96%", "ROC-AUC": 0.854, "Evaluation Context": "Standard spirometry input set"}
])
t6.to_csv(os.path.join(tables_dir, "table6_model_performance.csv"), index=False)

# Table 7: Literature Comparison
t7 = pd.DataFrame([
    {"Author & Year": "Amaral et al. (2018)", "Research Method": "Forced Oscillation Technique + SVM", "Dataset": "Local clinical cohort (N=86)", "Performance": "Accuracy: 88.4%", "Limitations": "Expensive oscillometry hardware, no IoT integration"},
    {"Author & Year": "Badr et al. (2020)", "Research Method": "Mobile Spirometry + Random Forest", "Dataset": "University hospital (N=120)", "Performance": "Sensitivity: 86.2%", "Limitations": "Manual calibration, no acoustic adventitious sound tracking"},
    {"Author & Year": "Rocha et al. (2019)", "Research Method": "ICBHI Respiratory Sound Challenge", "Dataset": "ICBHI Database (N=126)", "Performance": "Score: 64.5%", "Limitations": "Audio only; lacked spirometric volume integration"},
    {"Author & Year": "Tang et al. (2021)", "Research Method": "XGBoost for COPD Readmission", "Dataset": "MIMIC-III clinical records", "Performance": "AUC: 0.831", "Limitations": "Retrospective EHR data only; no point-of-care embedded sensor"},
    {"Author & Year": "This Work (2026)", "Research Method": "Dual-Modality (XGBoost + CRNN + ESP32)", "Dataset": "COPD Clinical (N=101) + ICBHI (N=126)", "Performance": "Accuracy: 97.39% / 84.15%", "Limitations": "Small clinical sample, cross-modal datasets evaluated in parallel"}
])
t7.to_csv(os.path.join(tables_dir, "table7_literature_comparison.csv"), index=False)

# Table 8: Proposed vs Existing Systems
t8 = pd.DataFrame([
    {"System Feature": "System Unit Cost", "Clinical Desktop Spirometer": "$2,000 - $8,000 USD", "Commercial Portable Spirometer": "$300 - $800 USD", "Proposed ESP32 System": "< $50 USD"},
    {"System Feature": "Acoustic Breath Auscultation", "Clinical Desktop Spirometer": "No (Requires separate stethoscope)", "Commercial Portable Spirometer": "No", "Proposed ESP32 System": "Yes (Simultaneous 16 kHz audio)"},
    {"System Feature": "Environmental BTPS Correction", "Clinical Desktop Spirometer": "Manual or internal barometer", "Commercial Portable Spirometer": "Rarely available", "Proposed ESP32 System": "Real-time via DHT22"},
    {"System Feature": "Machine Learning Integration", "Clinical Desktop Spirometer": "Rule-based cutoff only", "Commercial Portable Spirometer": "Basic linear thresholding", "Proposed ESP32 System": "XGBoost + CRNN + Late Fusion"},
    {"System Feature": "Telemedicine & WebSockets", "Clinical Desktop Spirometer": "Proprietary software / USB cable", "Commercial Portable Spirometer": "Proprietary Bluetooth app", "Proposed ESP32 System": "Open-standards WebSocket / WebAudio"}
])
t8.to_csv(os.path.join(tables_dir, "table8_proposed_vs_existing.csv"), index=False)

# Table 9: System Specifications
t9 = pd.DataFrame([
    {"Subsystem": "Embedded Controller", "Parameter": "SoC", "Engineering Specification": "ESP32 (Dual-core Xtensa 32-bit LX6 @ 240 MHz)"},
    {"Subsystem": "ADC Subsystem", "Parameter": "Resolution & Channels", "Engineering Specification": "12-bit SAR ADC (GPIO 34 = ADC1_CH6, GPIO 35 = ADC1_CH7)"},
    {"Subsystem": "Differential Pressure", "Parameter": "Transducer & Range", "Engineering Specification": "MPX10DP (0 to 10 kPa / 0 to 1.45 psi, 3.5 mV/kPa)"},
    {"Subsystem": "Signal Conditioning", "Parameter": "Instrumentation Amp", "Engineering Specification": "AD620 (CMRR > 100 dB, RG = 620 Ohm, Gain G ≈ 80)"},
    {"Subsystem": "Acoustic Sampling", "Parameter": "Audio Format", "Engineering Specification": "16 kHz Mono, 16-bit PCM, 128-bin Mel-spectrogram"},
    {"Subsystem": "Environmental Sensor", "Parameter": "Temperature & Humidity", "Engineering Specification": "DHT22 (-40 to 80 °C, 0 to 100% RH, single-bus digital)"},
    {"Subsystem": "Flow Sampling Rate", "Parameter": "Timer Interrupt Frequency", "Engineering Specification": "200 Hz (5 ms period, exceeding human respiratory bandwidth)"}
])
t9.to_csv(os.path.join(tables_dir, "table9_system_specifications.csv"), index=False)

# Table 10: Limitations and Future Improvements
t10 = pd.DataFrame([
    {"Research Aspect": "Clinical Sample Size", "Current Implementation Limitation": "N=101 patient cohort from single geographic location", "Proposed Enhancement / Roadmap": "Multi-center clinical trial across diverse patient demographics"},
    {"Research Aspect": "Multimodal Pairing", "Current Implementation Limitation": "Parallel training on disjoint clinical and acoustic datasets", "Proposed Enhancement / Roadmap": "Prospectively collect paired flow and acoustic recordings per patient"},
    {"Research Aspect": "Edge ML Execution", "Current Implementation Limitation": "Inference executed on Flask Python backend", "Proposed Enhancement / Roadmap": "Quantize models to TensorFlow Lite for Microcontrollers (ESP32 Edge)"},
    {"Research Aspect": "Mouthpiece Aerodynamics", "Current Implementation Limitation": "Requires benchtop 3.0 L syringe empirical calibration", "Proposed Enhancement / Roadmap": "Computational Fluid Dynamics (CFD) optimized 3D printed Venturi tube"},
    {"Research Aspect": "Imaging Integration", "Current Implementation Limitation": "Chest X-ray CNN currently conceptual", "Proposed Enhancement / Roadmap": "Incorporate DICOM thoracic imaging into centralized hospital EHR"}
])
t10.to_csv(os.path.join(tables_dir, "table10_limitations_future_work.csv"), index=False)

print("All 10 IEEE research tables generated in CSV format successfully!")
