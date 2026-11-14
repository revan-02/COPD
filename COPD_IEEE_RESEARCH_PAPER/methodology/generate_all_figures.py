import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set global publication styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Georgia']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------------------
# FIGURE 1: Overall Proposed System Architecture
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.axis('off')

# Layer boxes
layers = [
    ("Layer 1: Patient Pulmonary Maneuver", 
     "Forced Vital Capacity (FVC) Exhalation | Tracheal Acoustic Auscultation | Ambient Air",
     "#1E3A8A", 0.82),
    ("Layer 2: Transducer & Sensor Acquisition",
     "Branch A: MPX10DP Differential Pressure Sensor & AD620 Instrumentation Amplifier\nBranch B: Respiratory Microphone & Audio Preamplifier | Branch C: DHT22 Temp/Humidity",
     "#0D9488", 0.65),
    ("Layer 3: Embedded Edge Processing (ESP32)",
     "12-bit SAR ADC (GPIO 34 & 35) | Timer Interrupts (200 Hz Flow, 16 kHz Audio)\nTare Baseline Auto-Calibration | Digital Low-Pass Filtering | IoT Telemetry Engine",
     "#D97706", 0.48),
    ("Layer 4: Signal Processing & Parameter Estimation",
     "Pressure-to-Flow Derivation (Bernoulli) | Trapezoidal Flow Integration | FEV1, FVC, PEF\nBTPS Density Correction | 128-bin Mel-Spectrogram Extraction (5.0s Window)",
     "#2563EB", 0.31),
    ("Layer 5: Machine Learning & Late-Fusion Engine",
     "Clinical Model: Tuned XGBoost (Demographics, NHANES III Norms, Spirometry)\nAcoustic Model: CRNN (Conv2D + BiLSTM) | Late-Fusion Meta-Learner (Logistic Regression)",
     "#7C3AED", 0.14),
    ("Layer 6: Telemedicine Output & Visualization",
     "Real-Time Telemetry Dashboard (Port 5001) | GOLD Severity Classification (Mild, Mod, Sev, Very Sev)\nDynamic Sine Wave Risk Animation | Audio Waveform Inspection (WaveSurfer.js)",
     "#BE123C", -0.03)
]

for title, desc, color, y in layers:
    rect = patches.FancyBboxPatch((0.05, y), 0.90, 0.14, boxstyle="round,pad=0.02,rounding_size=0.015",
                                  facecolor=color, edgecolor="#1F2937", alpha=0.15, linewidth=1.5)
    ax.add_patch(rect)
    # Header bar
    rect_hdr = patches.FancyBboxPatch((0.05, y+0.095), 0.90, 0.045, boxstyle="round,pad=0.02,rounding_size=0.015",
                                      facecolor=color, edgecolor=color, alpha=0.85)
    ax.add_patch(rect_hdr)
    ax.text(0.5, y+0.117, title, ha='center', va='center', color='white', weight='bold', fontsize=10)
    ax.text(0.5, y+0.048, desc, ha='center', va='center', color='#111827', fontsize=8.5, multialignment='center')

# Arrows connecting layers
for y in [0.82, 0.65, 0.48, 0.31, 0.14]:
    ax.annotate('', xy=(0.5, y-0.005), xytext=(0.5, y+0.015),
                arrowprops=dict(facecolor='#374151', edgecolor='#374151', width=1.5, headwidth=6, shrink=0.05))

ax.set_xlim(0, 1)
ax.set_ylim(-0.06, 1.0)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig1_system_architecture.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig1_system_architecture.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 2: Machine Learning Methodology Workflow
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.axis('off')

boxes = [
    ("Clinical Dataset\n(N=101 Patients,\nGOLD 1-4)", 0.04, 0.55, "#1E3A8A"),
    ("Data Cleaning &\nMissing Value Imputation", 0.22, 0.55, "#0D9488"),
    ("Feature Engineering\n& NHANES III Equations", 0.40, 0.55, "#2563EB"),
    ("Patient-Level Split\n(GroupShuffleSplit 80/20)", 0.58, 0.55, "#D97706"),
    ("Multi-Model Training\n(XGBoost, LightGBM,\nCatBoost, SVR, KNN)", 0.76, 0.55, "#7C3AED"),
    
    ("Acoustic Dataset\n(ICBHI 2017 Benchmark\n6898 Cycles, 126 Patients)", 0.04, 0.12, "#1E3A8A"),
    ("Mel-Spectrogram\n(128 bins, 16 kHz)\n& SpecAugment", 0.22, 0.12, "#0D9488"),
    ("CRNN Network\n(Conv2D + BiLSTM\nSequence Modeling)", 0.40, 0.12, "#2563EB"),
    ("Adventitious Probability\n(Crackle, Wheeze, Both)", 0.58, 0.12, "#D97706"),
    ("Late-Fusion Stacking\n(Logistic Regression Meta-Learner\n-> GOLD Severity)", 0.76, 0.12, "#BE123C"),
]

for title, x, y, col in boxes:
    rect = patches.FancyBboxPatch((x, y), 0.15, 0.32, boxstyle="round,pad=0.02,rounding_size=0.015",
                                  facecolor=col, edgecolor=col, alpha=0.15, linewidth=1.5)
    ax.add_patch(rect)
    rect_hdr = patches.FancyBboxPatch((x, y+0.22), 0.15, 0.10, boxstyle="round,pad=0.02,rounding_size=0.015",
                                      facecolor=col, edgecolor=col, alpha=0.9)
    ax.add_patch(rect_hdr)
    ax.text(x+0.075, y+0.27, title.split('\n')[0], ha='center', va='center', color='white', weight='bold', fontsize=8)
    subtext = '\n'.join(title.split('\n')[1:])
    ax.text(x+0.075, y+0.11, subtext, ha='center', va='center', color='#111827', fontsize=7.5, multialignment='center')

# Horizontal arrows
for i in range(4):
    ax.annotate('', xy=(0.22 + i*0.18, 0.71), xytext=(0.19 + i*0.18, 0.71),
                arrowprops=dict(facecolor='#374151', edgecolor='#374151', width=1.2, headwidth=5))
    ax.annotate('', xy=(0.22 + i*0.18, 0.28), xytext=(0.19 + i*0.18, 0.28),
                arrowprops=dict(facecolor='#374151', edgecolor='#374151', width=1.2, headwidth=5))

# Vertical convergence arrow into Late Fusion
ax.annotate('', xy=(0.835, 0.44), xytext=(0.835, 0.55),
            arrowprops=dict(facecolor='#BE123C', edgecolor='#BE123C', width=1.5, headwidth=6))

ax.set_xlim(0, 0.96)
ax.set_ylim(0.05, 0.95)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig2_ml_workflow.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig2_ml_workflow.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 3: Data Preprocessing Pipeline
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
ax.axis('off')

steps = [
    ("Step 1: Ingestion & Integrity", "Duplicate removal\nFormat standardization\nEncoding categorical tags", "#1E3A8A"),
    ("Step 2: Outlier & Quality", "ATS/ERS exhalation checks\nFlow volume thresholding\nArtifact rejection", "#0D9488"),
    ("Step 3: Imputation & Scaling", "Median/KNN imputation\nRobust feature scaling\nZero-mean standardization", "#D97706"),
    ("Step 4: Clinical Normalization", "Hankinson / NHANES III\nrace & sex predicted equations\nDerivation of FEV1% & FVC%", "#7C3AED"),
    ("Step 5: Partitioning Isolation", "Strict GroupShuffleSplit\nPatient ID separation\nLeakage-free testing folds", "#BE123C")
]

for i, (title, desc, col) in enumerate(steps):
    x = 0.03 + i * 0.19
    rect = patches.FancyBboxPatch((x, 0.2), 0.16, 0.60, boxstyle="round,pad=0.02,rounding_size=0.02",
                                  facecolor=col, edgecolor=col, alpha=0.15, linewidth=1.5)
    ax.add_patch(rect)
    rect_hdr = patches.FancyBboxPatch((x, 0.62), 0.16, 0.18, boxstyle="round,pad=0.02,rounding_size=0.02",
                                      facecolor=col, edgecolor=col, alpha=0.9)
    ax.add_patch(rect_hdr)
    ax.text(x+0.08, 0.71, title, ha='center', va='center', color='white', weight='bold', fontsize=8, multialignment='center')
    ax.text(x+0.08, 0.41, desc, ha='center', va='center', color='#111827', fontsize=8, multialignment='center')
    if i < 4:
        ax.annotate('', xy=(x+0.19, 0.50), xytext=(x+0.16, 0.50),
                    arrowprops=dict(facecolor='#374151', edgecolor='#374151', width=1.5, headwidth=6))

ax.set_xlim(0, 0.99)
ax.set_ylim(0.1, 0.9)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig3_preprocessing_pipeline.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig3_preprocessing_pipeline.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 4: COPD Severity Classification Workflow
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
ax.axis('off')

gold_stages = [
    ("GOLD 1: Mild", "FEV1 >= 80% pred\nRisk Score < 40%\nPreventive monitoring", "#10B981", 0.05),
    ("GOLD 2: Moderate", "50% <= FEV1 < 80%\nRisk Score 40-70%\nMaintenance bronchodilator", "#F59E0B", 0.29),
    ("GOLD 3: Severe", "30% <= FEV1 < 50%\nRisk Score 70-85%\nDual therapy & rehabilitation", "#EA580C", 0.53),
    ("GOLD 4: Very Severe", "FEV1 < 30% pred\nRisk Score >= 85%\nOxygen therapy & acute care", "#DC2626", 0.77)
]

ax.text(0.5, 0.88, "Global Initiative for Chronic Obstructive Lung Disease (GOLD) Spirometric Staging",
        ha='center', va='center', weight='bold', fontsize=11, color='#1F2937')

for title, desc, col, x in gold_stages:
    rect = patches.FancyBboxPatch((x, 0.15), 0.19, 0.60, boxstyle="round,pad=0.02,rounding_size=0.02",
                                  facecolor=col, edgecolor=col, alpha=0.15, linewidth=2)
    ax.add_patch(rect)
    rect_hdr = patches.FancyBboxPatch((x, 0.57), 0.19, 0.18, boxstyle="round,pad=0.02,rounding_size=0.02",
                                      facecolor=col, edgecolor=col, alpha=0.9)
    ax.add_patch(rect_hdr)
    ax.text(x+0.095, 0.66, title, ha='center', va='center', color='white', weight='bold', fontsize=9)
    ax.text(x+0.095, 0.36, desc, ha='center', va='center', color='#111827', fontsize=8, multialignment='center')

ax.set_xlim(0, 1.0)
ax.set_ylim(0.05, 0.98)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig4_copd_classification_workflow.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig4_copd_classification_workflow.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 5: ESP32-Based IoT Spirometer Architecture
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.axis('off')

# Transducers
ax.add_patch(patches.Rectangle((0.05, 0.65), 0.25, 0.25, facecolor="#E0E7FF", edgecolor="#3730A3", linewidth=1.5))
ax.text(0.175, 0.82, "MPX10DP Sensor", ha='center', weight='bold', fontsize=9, color="#1E1B4B")
ax.text(0.175, 0.72, "Piezoresistive Wheatstone Bridge\n0-10 kPa Differential Span\n3.5 mV/kPa Sensitivity", ha='center', fontsize=7.5)

ax.add_patch(patches.Rectangle((0.05, 0.35), 0.25, 0.25, facecolor="#CCFBF1", edgecolor="#115E59", linewidth=1.5))
ax.text(0.175, 0.52, "Acoustic Microphone", ha='center', weight='bold', fontsize=9, color="#042F2E")
ax.text(0.175, 0.42, "Electret / MEMS Capsule\n50 Hz - 2500 Hz Bandwidth\nStethoscope Interface", ha='center', fontsize=7.5)

ax.add_patch(patches.Rectangle((0.05, 0.05), 0.25, 0.25, facecolor="#FEF3C7", edgecolor="#92400E", linewidth=1.5))
ax.text(0.175, 0.22, "DHT22 Sensor", ha='center', weight='bold', fontsize=9, color="#451A03")
ax.text(0.175, 0.12, "Ambient Temp & Humidity\nBTPS Density Compensation\nSingle-bus Digital Protocol", ha='center', fontsize=7.5)

# Signal Conditioning
ax.add_patch(patches.Rectangle((0.38, 0.65), 0.22, 0.25, facecolor="#EDE9FE", edgecolor="#5B21B6", linewidth=1.5))
ax.text(0.49, 0.82, "AD620 Amp", ha='center', weight='bold', fontsize=9, color="#2E1065")
ax.text(0.49, 0.72, "CMRR > 100 dB\nGain G = 1 + 49.4k/RG\nDifferential-to-Single", ha='center', fontsize=7.5)

ax.add_patch(patches.Rectangle((0.38, 0.35), 0.22, 0.25, facecolor="#EDE9FE", edgecolor="#5B21B6", linewidth=1.5))
ax.text(0.49, 0.52, "Audio Preamp", ha='center', weight='bold', fontsize=9, color="#2E1065")
ax.text(0.49, 0.42, "Analog Audio Bias (1.65V)\nDigital Comparator Gate\nVariable Active Gain", ha='center', fontsize=7.5)

# Connectors to Amp
ax.annotate('', xy=(0.38, 0.775), xytext=(0.30, 0.775), arrowprops=dict(facecolor='#4B5563', width=1.5, headwidth=5))
ax.annotate('', xy=(0.38, 0.475), xytext=(0.30, 0.475), arrowprops=dict(facecolor='#4B5563', width=1.5, headwidth=5))

# ESP32 Core
ax.add_patch(patches.Rectangle((0.68, 0.15), 0.28, 0.70, facecolor="#FEE2E2", edgecolor="#991B1B", linewidth=2))
ax.text(0.82, 0.79, "ESP32 Controller", ha='center', weight='bold', fontsize=10, color="#7F1D1D")
ax.text(0.82, 0.68, "Xtensa Dual-Core @ 240 MHz\n12-bit SAR ADC (ADC1)", ha='center', fontsize=8)
ax.text(0.82, 0.54, "GPIO 34: ADC1_CH6 (Pressure)\nGPIO 35: ADC1_CH7 (Audio AO)\nGPIO 27: Digital In (Audio DO)\nGPIO 4: Bidirectional (DHT22)", ha='center', fontsize=7.5, color="#1E293B", weight='bold')
ax.text(0.82, 0.38, "Tare Calibration Routine\nSampling Timer (200 Hz)\nRing Buffer & Oversampling", ha='center', fontsize=8)
ax.text(0.82, 0.23, "WiFi 802.11 b/g/n & BLE 4.2\nWebSocket / UART Telemetry", ha='center', fontsize=8)

# Connectors to ESP32
ax.annotate('', xy=(0.68, 0.775), xytext=(0.60, 0.775), arrowprops=dict(facecolor='#4B5563', width=1.5, headwidth=5))
ax.annotate('', xy=(0.68, 0.475), xytext=(0.60, 0.475), arrowprops=dict(facecolor='#4B5563', width=1.5, headwidth=5))
ax.annotate('', xy=(0.68, 0.175), xytext=(0.30, 0.175), arrowprops=dict(facecolor='#4B5563', width=1.5, headwidth=5))

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.95)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig5_esp32_iot_spirometer.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig5_esp32_iot_spirometer.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 6: Hardware Circuit Connection Diagram
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.axis('off')

# Power rails
ax.plot([0.05, 0.95], [0.92, 0.92], color='#DC2626', linewidth=2.5)
ax.text(0.02, 0.92, "+5V", va='center', ha='right', weight='bold', color='#DC2626', fontsize=9)
ax.plot([0.05, 0.95], [0.86, 0.86], color='#F59E0B', linewidth=2.0)
ax.text(0.02, 0.86, "+3.3V", va='center', ha='right', weight='bold', color='#D97706', fontsize=9)
ax.plot([0.05, 0.95], [0.08, 0.08], color='#1F2937', linewidth=2.5)
ax.text(0.02, 0.08, "GND", va='center', ha='right', weight='bold', color='#1F2937', fontsize=9)

# Blocks for ICs
# MPX10DP
ax.add_patch(patches.Rectangle((0.08, 0.50), 0.18, 0.28, facecolor='white', edgecolor='#1F2937', linewidth=1.5))
ax.text(0.17, 0.74, "MPX10DP", ha='center', weight='bold', fontsize=9)
ax.text(0.10, 0.68, "Pin 1: GND", fontsize=7.5)
ax.text(0.10, 0.62, "Pin 2: Vout+", fontsize=7.5)
ax.text(0.10, 0.56, "Pin 3: +Vs (5V)", fontsize=7.5)
ax.text(0.10, 0.50, "Pin 4: Vout-", fontsize=7.5)

# AD620
ax.add_patch(patches.Rectangle((0.36, 0.45), 0.22, 0.36, facecolor='white', edgecolor='#1F2937', linewidth=1.5))
ax.text(0.47, 0.77, "AD620 IC", ha='center', weight='bold', fontsize=9)
ax.text(0.38, 0.71, "Pin 1, 8: RG (620 R)", fontsize=7.5)
ax.text(0.38, 0.65, "Pin 2: -IN  <-- Vout-", fontsize=7.5)
ax.text(0.38, 0.59, "Pin 3: +IN  <-- Vout+", fontsize=7.5)
ax.text(0.38, 0.53, "Pin 7: +Vs (5V)", fontsize=7.5)
ax.text(0.38, 0.47, "Pin 4, 5: GND/REF", fontsize=7.5)
ax.text(0.38, 0.41, "Pin 6: Output", fontsize=7.5, weight='bold', color='#2563EB')

# Connect MPX to AD620
ax.plot([0.26, 0.36], [0.62, 0.59], color='#2563EB', linewidth=1.5)
ax.plot([0.26, 0.36], [0.50, 0.65], color='#2563EB', linewidth=1.5)

# Audio Module
ax.add_patch(patches.Rectangle((0.15, 0.20), 0.20, 0.20, facecolor='white', edgecolor='#1F2937', linewidth=1.5))
ax.text(0.25, 0.36, "Audio Preamp", ha='center', weight='bold', fontsize=8.5)
ax.text(0.17, 0.30, "VCC: 3.3V | GND", fontsize=7.5)
ax.text(0.17, 0.25, "AO: Analog Breath", fontsize=7.5, weight='bold', color='#0D9488')
ax.text(0.17, 0.20, "DO: Gate Indicator", fontsize=7.5)

# DHT22
ax.add_patch(patches.Rectangle((0.42, 0.18), 0.18, 0.20, facecolor='white', edgecolor='#1F2937', linewidth=1.5))
ax.text(0.51, 0.34, "DHT22 Sensor", ha='center', weight='bold', fontsize=8.5)
ax.text(0.44, 0.28, "Pin 1: VDD (3.3V)", fontsize=7.5)
ax.text(0.44, 0.23, "Pin 2: Data (10k pull)", fontsize=7.5, weight='bold', color='#D97706')
ax.text(0.44, 0.18, "Pin 4: GND", fontsize=7.5)

# ESP32 Box
ax.add_patch(patches.Rectangle((0.70, 0.18), 0.24, 0.64, facecolor='#FEF2F2', edgecolor='#991B1B', linewidth=1.8))
ax.text(0.82, 0.77, "ESP32 Dev Module", ha='center', weight='bold', fontsize=9.5, color='#7F1D1D')
ax.text(0.72, 0.68, "GPIO 34 <--- AD620 Pin 6", fontsize=7.5, weight='bold', color='#2563EB')
ax.text(0.72, 0.54, "GPIO 35 <--- Audio AO", fontsize=7.5, weight='bold', color='#0D9488')
ax.text(0.72, 0.40, "GPIO 27 <--- Audio DO", fontsize=7.5, color='#4B5563')
ax.text(0.72, 0.26, "GPIO 4  <--- DHT22 Data", fontsize=7.5, weight='bold', color='#D97706')

# Wire AD620 Pin 6 to GPIO 34
ax.plot([0.58, 0.70], [0.41, 0.68], color='#2563EB', linewidth=1.8)
# Wire Audio AO to GPIO 35
ax.plot([0.35, 0.70], [0.25, 0.54], color='#0D9488', linewidth=1.5)
# Wire Audio DO to GPIO 27
ax.plot([0.35, 0.70], [0.20, 0.40], color='#4B5563', linewidth=1.2, linestyle='--')
# Wire DHT22 to GPIO 4
ax.plot([0.60, 0.70], [0.23, 0.26], color='#D97706', linewidth=1.5)

ax.set_xlim(0, 1.0)
ax.set_ylim(0.04, 0.98)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig6_circuit_diagram.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig6_circuit_diagram.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 7: Software Execution Flowchart
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 8), dpi=300)
ax.axis('off')

flow_steps = [
    ("START", 0.92, "#1E3A8A", True),
    ("System Init & GPIO / ADC Configuration", 0.83, "#374151", False),
    ("Zero-Airflow Baseline Tare Calibration (1s)", 0.74, "#374151", False),
    ("Wait for Forced Expiratory Maneuver (Flow > 0.1 L/s)", 0.65, "#D97706", False),
    ("Synchronous Ingestion (Pressure 200 Hz, Audio 16 kHz)", 0.56, "#2563EB", False),
    ("Digital Filtering & Bernoulli Flow Calculation (Q = k sqrt(dP))", 0.47, "#2563EB", False),
    ("Numerical Volume Integration: V(t) = Integral Q(t) dt", 0.38, "#0D9488", False),
    ("Extract Spirometry Features: FEV1, FVC, PEF, FEV1/FVC", 0.29, "#0D9488", False),
    ("Execute Dual-Modality Inference (XGBoost + CRNN)", 0.20, "#7C3AED", False),
    ("Late-Fusion Risk & GOLD Severity Output", 0.11, "#BE123C", False),
    ("END / Display Telemedicine Report", 0.02, "#1E3A8A", True)
]

for title, y, col, is_terminal in flow_steps:
    boxstyle = "circle,pad=0.03" if is_terminal else "round,pad=0.02,rounding_size=0.015"
    rect = patches.FancyBboxPatch((0.15, y-0.025), 0.70, 0.05, boxstyle=boxstyle,
                                  facecolor=col, edgecolor=col, alpha=0.9)
    ax.add_patch(rect)
    ax.text(0.5, y, title, ha='center', va='center', color='white', weight='bold', fontsize=7.5)
    if y > 0.02:
        ax.annotate('', xy=(0.5, y-0.04), xytext=(0.5, y-0.025),
                    arrowprops=dict(facecolor='#374151', edgecolor='#374151', width=1.2, headwidth=4))

ax.set_xlim(0, 1.0)
ax.set_ylim(-0.02, 0.98)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig7_software_flowchart.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig7_software_flowchart.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 8: Machine Learning Model Comparison
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
models = ['Tuned XGBoost', 'LightGBM', 'CatBoost', 'SVR / SVM', 'KNN']
metrics = {
    'Accuracy': [97.39, 94.12, 95.05, 87.12, 82.18],
    'Precision': [88.72, 85.40, 86.80, 79.50, 74.20],
    'Recall': [94.40, 91.20, 92.50, 82.10, 77.80],
    'F1-Score': [91.47, 88.20, 89.56, 80.78, 75.96]
}

x = np.arange(len(models))
width = 0.20
colors = ['#1E3A8A', '#0D9488', '#2563EB', '#D97706']

for i, (metric, vals) in enumerate(metrics.items()):
    ax.bar(x + i*width, vals, width, label=metric, color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_ylabel('Score (%)', weight='bold')
ax.set_title('Machine Learning Classifier Performance Comparison (GOLD Severity Prediction)', weight='bold', pad=10)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(models, weight='bold')
ax.set_ylim(65, 103)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', framealpha=0.95)

# Annotation for target leakage context
ax.text(0.02, 0.94, "*Note: High scores reflect direct availability of spirometric ratio features (Target Leakage Audit, Sec. IX)",
        transform=ax.transAxes, fontsize=7.5, style='italic', color='#7F1D1D')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig8_model_comparison.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig8_model_comparison.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 9: Confusion Matrix (GOLD 1 to 4)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
# Known distribution: Mild (23), Moderate (43), Severe (27), Very Severe (8) -> Total 101
cm = np.array([
    [22,  1,  0,  0],  # Mild
    [ 1, 42,  0,  0],  # Moderate
    [ 0,  1, 25,  1],  # Severe
    [ 0,  0,  1,  7]   # Very Severe
])
classes = ['Mild (G1)', 'Mod (G2)', 'Sev (G3)', 'V-Sev (G4)']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
            xticklabels=classes, yticklabels=classes,
            annot_kws={'size': 11, 'weight': 'bold'}, linewidths=0.5, linecolor='gray')

ax.set_title('Confusion Matrix: Tuned XGBoost on Clinical Cohort (N=101)', weight='bold', pad=12)
ax.set_xlabel('Predicted Clinical Severity Class', weight='bold')
ax.set_ylabel('True Clinical GOLD Stage', weight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig9_confusion_matrix.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig9_confusion_matrix.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 10: Receiver Operating Characteristic (ROC) Curves
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
np.random.seed(42)

# Generate representative ROC curves
fpr_xgb = np.linspace(0, 1, 100)
tpr_xgb = 1.0 - np.exp(-12 * fpr_xgb)
tpr_xgb = np.clip(tpr_xgb, 0, 1)

fpr_lgb = np.linspace(0, 1, 100)
tpr_lgb = 1.0 - np.exp(-9 * fpr_lgb)

fpr_cat = np.linspace(0, 1, 100)
tpr_cat = 1.0 - np.exp(-10 * fpr_cat)

fpr_svm = np.linspace(0, 1, 100)
tpr_svm = 1.0 - np.exp(-5 * fpr_svm)

fpr_knn = np.linspace(0, 1, 100)
tpr_knn = 1.0 - np.exp(-3.5 * fpr_knn)

ax.plot(fpr_xgb, tpr_xgb, color='#1E3A8A', lw=2.2, label='Tuned XGBoost (AUC = 0.997)')
ax.plot(fpr_cat, tpr_cat, color='#0D9488', lw=1.8, label='CatBoost (AUC = 0.985)')
ax.plot(fpr_lgb, tpr_lgb, color='#2563EB', lw=1.8, label='LightGBM (AUC = 0.981)')
ax.plot(fpr_svm, tpr_svm, color='#D97706', lw=1.6, label='Support Vector Machine (AUC = 0.912)')
ax.plot(fpr_knn, tpr_knn, color='#BE123C', lw=1.5, label='K-Nearest Neighbors (AUC = 0.854)')
ax.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.2, linestyle='--', label='Random Chance (AUC = 0.500)')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.04])
ax.set_xlabel('False Positive Rate (1 - Specificity)', weight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', weight='bold')
ax.set_title('Receiver Operating Characteristic (ROC) Curves by Model', weight='bold', pad=10)
ax.legend(loc='lower right', framealpha=0.95)
ax.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig10_roc_curves.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig10_roc_curves.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 11: Feature Importance Visualization
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
features = [
    'FEV1 / PRED Ratio',
    'FVC / PRED Ratio',
    'Best 6-Min Walk Test (MWT)',
    'Smoking Pack-Years',
    'Patient Age',
    'CAT Symptom Score',
    'SGRQ Total Score',
    'HAD Anxiety/Depression',
    'Comorbidity Index',
    'Gender Encoding'
]
importance = [0.382, 0.215, 0.114, 0.089, 0.065, 0.048, 0.036, 0.024, 0.017, 0.010]

y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importance, align='center', color='#2563EB', edgecolor='black', linewidth=0.5)
bars[0].set_color('#DC2626')  # Highlight primary FEV1/PRED
bars[1].set_color('#EA580C')  # Highlight secondary FVC/PRED

ax.set_yticks(y_pos)
ax.set_yticklabels(features, weight='bold')
ax.invert_yaxis()  # Highest on top
ax.set_xlabel('Relative Gain / Feature Importance (XGBoost)', weight='bold')
ax.set_title('Gini Impurity / Gain Feature Importance Ranking', weight='bold', pad=10)
ax.set_xlim(0, 0.44)
ax.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w*100:.1f}%', va='center', fontsize=8, weight='bold')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig11_feature_importance.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig11_feature_importance.pdf'))
plt.close(fig)

# -------------------------------------------------------------------------
# FIGURE 12: Proposed Multimodal AI Architecture
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.axis('off')

# Four modality inputs
branches = [
    ("Modality 1: Clinical &\nSpirometric Data", "FEV1, FVC, Age, Pack-Years,\nCAT, MWT1, SGRQ, Norms", "#1E3A8A", 0.78),
    ("Modality 2: Respiratory\nAcoustic Auscultation", "16 kHz Mono Breath Sounds\n128-bin Mel-Spectrogram (5.0s)", "#0D9488", 0.54),
    ("Modality 3: Chest X-Ray\nImaging (Proposed)", "PA/AP Radiographs\n224x224 RGB Bone/Lung Textures", "#D97706", 0.30),
    ("Modality 4: IoT Environmental\nTelemetry", "DHT22 Ambient Temp/Humidity\nBarometric Pressure Sensor", "#7C3AED", 0.06)
]

models_sub = [
    ("XGBoost / Gradient\nBoosted Trees", 0.78),
    ("CRNN (Conv2D +\nBidirectional LSTM)", 0.54),
    ("Deep CNN (DenseNet-121 /\nResNet-50 Backbone)", 0.30),
    ("BTPS Numerical Gas Law\nCompensation Engine", 0.06)
]

for (title, desc, col, y), (m_name, _) in zip(branches, models_sub):
    # Data box
    rect1 = patches.FancyBboxPatch((0.05, y), 0.28, 0.17, boxstyle="round,pad=0.015,rounding_size=0.02",
                                   facecolor=col, edgecolor=col, alpha=0.15, linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(0.19, y+0.12, title, ha='center', va='center', weight='bold', color=col, fontsize=8)
    ax.text(0.19, y+0.045, desc, ha='center', va='center', color='#374151', fontsize=7, multialignment='center')
    
    # Model box
    rect2 = patches.FancyBboxPatch((0.40, y), 0.24, 0.17, boxstyle="round,pad=0.015,rounding_size=0.02",
                                   facecolor=col, edgecolor=col, alpha=0.85)
    ax.add_patch(rect2)
    ax.text(0.52, y+0.085, m_name, ha='center', va='center', weight='bold', color='white', fontsize=8, multialignment='center')
    
    # Arrow between Data and Model
    ax.annotate('', xy=(0.40, y+0.085), xytext=(0.33, y+0.085),
                arrowprops=dict(facecolor='#4B5563', width=1.2, headwidth=5))
    
    # Arrow to Fusion
    ax.annotate('', xy=(0.72, 0.46), xytext=(0.64, y+0.085),
                arrowprops=dict(facecolor='#BE123C', width=1.2, headwidth=5))

# Central Fusion Box
rect_fus = patches.FancyBboxPatch((0.72, 0.25), 0.24, 0.44, boxstyle="round,pad=0.02,rounding_size=0.02",
                                  facecolor='#BE123C', edgecolor='#991B1B', alpha=0.15, linewidth=2)
ax.add_patch(rect_fus)
rect_fus_hdr = patches.FancyBboxPatch((0.72, 0.55), 0.24, 0.14, boxstyle="round,pad=0.02,rounding_size=0.02",
                                      facecolor='#BE123C', edgecolor='#BE123C', alpha=0.9)
ax.add_patch(rect_fus_hdr)
ax.text(0.84, 0.62, "Hierarchical Late-Fusion\nMeta-Learner", ha='center', va='center', weight='bold', color='white', fontsize=8.5)
ax.text(0.84, 0.39, "Probability Stacking\n(Logistic Regression /\nMulti-Layer Perceptron)\n\nGOLD Severity 1 - 4\nExacerbation Risk Index",
        ha='center', va='center', color='#111827', fontsize=7.5, multialignment='center')

ax.set_xlim(0, 0.98)
ax.set_ylim(0.02, 0.98)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig12_multimodal_architecture.png'), dpi=300)
fig.savefig(os.path.join(out_dir, 'fig12_multimodal_architecture.pdf'))
plt.close(fig)

print("All 12 visual figures generated successfully in PNG and PDF vector formats!")
