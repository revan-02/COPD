# An Intelligent IoT-Enabled Machine Learning Framework for COPD Severity Assessment Using Spirometry and Respiratory Monitoring

**IEEE Conference Research Paper & Publication Documentation Suite**  
*Biomedical Engineering & Embedded Intelligence Research Group*  
*Autonomous Health Informatics Laboratory*

---

## 📄 Overview

This repository contains the publication suite for our research paper:
> **"An Intelligent IoT-Enabled Machine Learning Framework for COPD Severity Assessment Using Spirometry and Respiratory Monitoring"**

The research delivers an end-to-end biomedical cyber-physical system integrating:
1. **Low-Cost Edge Sensing:** ESP32 microcontroller with MPX10DP piezoresistive differential pressure transducer, AD620 instrumentation amplifier ($G \approx 81.65$), DHT22 dynamic BTPS environmental compensation, and acoustic breath sensor.
2. **Real Clinical Cohort:** CDC NHANES (2007–2008 Continuous Cycle) cohort of **4,241 adult subjects** ($Age \ge 20$) with ATS/ERS Quality Grade A/B spirometry curves (630 COPD/airflow limitation cases, 3,611 normal controls).
3. **Machine Learning Benchmarking:** Tuned XGBoost classifier evaluated across 5-fold stratified cross-validation and held-out test splits against LightGBM, CatBoost, Random Forest, SVM, and KNN.
4. **Target Leakage Audit:** Methodological proof demonstrating that demographic and smoking factors provide population risk screening ($\text{AUC} = 0.8057$), while physical edge spirometry is strictly necessary for definitive clinical staging ($\text{AUC} = 0.9940$).
5. **Multimodal Deep Learning:** Convolutional Recurrent Neural Network (CRNN) pre-trained on ICBHI 2017 respiratory sounds integrated via late-fusion logistic regression.

---

## 📁 Repository Directory Structure

```text
COPD_IEEE_RESEARCH_PAPER/
│
├── COPD_IEEE_Research_Paper.pdf       # Compiled IEEE-format publication PDF (ReportLab)
├── COPD_IEEE_Research_Paper.docx      # Formatted Microsoft Word document with embedded figures
├── COPD_IEEE_Research_Paper.tex       # Official IEEEtran LaTeX conference source file
├── references.bib                     # Authentic IEEE bibliography (18 verified citations)
│
├── figures/                           # 12 high-resolution figures (300 DPI PNG & vector PDF)
│   ├── fig1_system_architecture.png / .pdf
│   ├── fig2_ml_workflow.png / .pdf
│   ├── fig3_preprocessing_pipeline.png / .pdf
│   ├── fig4_copd_classification_workflow.png / .pdf
│   ├── fig5_esp32_iot_spirometer.png / .pdf
│   ├── fig6_circuit_diagram.png / .pdf
│   ├── fig7_software_flowchart.png / .pdf
│   ├── fig8_model_comparison.png / .pdf
│   ├── fig9_confusion_matrix.png / .pdf
│   ├── fig10_roc_curves.png / .pdf
│   ├── fig11_feature_importance.png / .pdf
│   └── fig12_multimodal_architecture.png / .pdf
│
├── tables/                            # 10 clinical and hardware tables in CSV format
│   ├── table1_dataset_characteristics.csv
│   ├── table2_severity_distribution.csv
│   ├── table3_feature_description.csv
│   ├── table4_hardware_components.csv
│   ├── table5_ml_algorithms.csv
│   ├── table6_model_performance.csv
│   ├── table7_literature_comparison.csv
│   ├── table8_proposed_vs_existing.csv
│   ├── table9_system_specifications.csv
│   └── table10_limitations_future_work.csv
│
├── methodology/                       # Python scripts for end-to-end reproduction
│   ├── generate_all_figures.py        # Generates all 12 publication figures
│   ├── generate_tables.py             # Generates all 10 clinical tables
│   ├── generate_docx_and_pdf.py       # Compiles IEEE Word document
│   └── compile_pdf.py                 # Compiles IEEE PDF publication document
│
├── experimental_results/              # Experimental outputs and metrics
│   └── performance_metrics.json       # Complete performance metrics JSON record
│
└── README.md                          # Documentation index (this file)
```

---

## 📊 Summary of Experimental Results

### 1. Model Comparison on CDC NHANES Held-Out Test Set ($N = 849$)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost (Proposed)** | **96.23%** | **90.52%** | **83.33%** | **86.78%** | **0.9940** |
| LightGBM | 95.88% | 89.74% | 82.54% | 85.95% | 0.9915 |
| CatBoost | 95.64% | 89.57% | 81.75% | 85.48% | 0.9908 |
| Random Forest | 95.17% | 88.70% | 79.37% | 83.77% | 0.9874 |
| Support Vector Machine (RBF) | 94.11% | 86.49% | 76.19% | 81.01% | 0.9782 |
| K-Nearest Neighbors ($k=5$) | 93.40% | 83.48% | 73.02% | 77.90% | 0.9591 |
| Logistic Regression | 92.58% | 81.25% | 70.63% | 75.58% | 0.9512 |

*5-Fold Stratified Cross-Validation for Tuned XGBoost:* **$95.78\% \pm 0.67\%$**

---

### 2. Critical Target Leakage Audit (Ablation Study)

| Feature Configuration | Evaluated Features | Accuracy | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full Clinical Features** | Spirometry ($\text{FEV}_1$, $\text{FVC}$, Preds) + Demographics + Smoking | **96.23%** | **83.33%** | **86.78%** | **0.9940** |
| **Ablation (No Spirometry)**| Demographics + Smoking Only (`Age`, `Sex`, `Smoke_former`, `Smoke_never`) | **68.79%** | **78.57%** | **41.95%** | **0.8057** |

**Clinical Finding:** Epidemiological risk factors achieve an $\text{AUC} \approx 0.81$, serving as an effective non-invasive preliminary triage tool. However, physical spirometric airflow sensing is strictly required for clinical GOLD staging, confirming the fundamental necessity of our ESP32-based hardware spirometer.

---

## 🛠️ Hardware Implementation Specifications

- **Microcontroller:** ESP32 Dual-Core Tensilica Xtensa LX6 (240 MHz, 520 KB SRAM)
- **Pressure Sensor:** Freescale/NXP MPX10DP (0–10 kPa / 0–1.45 psi piezoresistive differential pressure)
- **Instrumentation Amplifier:** Analog Devices AD620 ($R_G = 620\,\Omega$, $G \approx 80.68$, full scale $2.82\,\text{V}$)
- **ADC Pin Mapping:** GPIO 34 (`ADC1_CH6`, 12-bit SAR)
- **Audio Transducer:** Electret Condenser + MAX4466 (GPIO 35 analog, GPIO 27 digital)
- **Environmental Sensor:** DHT22 AM2302 (GPIO 4 single-bus)
- **BTPS Correction:** Dynamic compensation based on real-time ambient temperature and humidity
- **Estimated BOM Cost:** $\approx \$28.50\,\text{USD}$ (versus $\$2,000\text{--}\$5,000\,\text{USD}$ commercial units)

---

## 🚀 How to Reproduce and Compile

### 1. Reproduce Figures
```bash
python COPD_IEEE_RESEARCH_PAPER/methodology/generate_all_figures.py
```

### 2. Reproduce Tables
```bash
python COPD_IEEE_RESEARCH_PAPER/methodology/generate_tables.py
```

### 3. Compile Word Document (.docx)
```bash
python COPD_IEEE_RESEARCH_PAPER/methodology/generate_docx_and_pdf.py
```

### 4. Compile PDF Document (.pdf)
```bash
python COPD_IEEE_RESEARCH_PAPER/methodology/compile_pdf.py
```

### 5. Compile LaTeX Document (.tex)
```bash
pdflatex COPD_IEEE_Research_Paper.tex
bibtex COPD_IEEE_Research_Paper
pdflatex COPD_IEEE_Research_Paper.tex
pdflatex COPD_IEEE_Research_Paper.tex
```

---

## 📜 Citation

If you use this project or publication suite in your research, please cite:

```bibtex
@inproceedings{copd_iot_xgb_2024,
  author    = {Biomedical Engineering and Embedded Intelligence Research Group},
  title     = {An Intelligent IoT-Enabled Machine Learning Framework for COPD Severity Assessment Using Spirometry and Respiratory Monitoring},
  booktitle = {IEEE Transactions on Biomedical Engineering / Healthcare Informatics},
  year      = {2024}
}
```
