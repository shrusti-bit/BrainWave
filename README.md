# 🧠 MindWave BCI

Interactive MindWave BCI platform with real-time EEG processing and cognitive state classification using Python and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

MindWave BCI is a complete end-to-end Brain-Computer Interface that simulates multi-channel EEG, extracts physiologically meaningful features, trains multiple machine learning models, and performs real-time cognitive state classification in a polished Streamlit app.

- 61 EEG channels simulated at 250 Hz
- 4 cognitive states: Resting, Memory, Math, Attention
- High-dimensional feature engineering (power, relative power, spectral shape, time stats)
- Three ML models trained and evaluated (Random Forest, SVM, Logistic Regression)
- Real-time demo: live generation, prediction, confidence, plots, feature importance

---

## 🚀 Quick Start

```bash
pip install -r requirements_final.txt
streamlit run final_mindwave.py
```

Then use the app:
1) Click “🚀 Train Models”
2) Click “🎯 Generate New EEG Sample”
3) View prediction, confidence, EEG plots, and feature importance

---

## 🧪 Data (Simulated EEG)

The system uses simulated EEG so it runs anywhere without hardware or datasets.

- Channels: 61
- Sampling rate: 250 Hz
- Duration per sample: 5 s
- States and their simulated logic:
  - Resting: dominant alpha rhythm (≈10 Hz) + noise
  - Memory: alpha base + gamma (≈40 Hz) bursts
  - Math: beta (≈20 Hz) + brief transient spikes
  - Attention: mixture of theta (≈6 Hz) and beta (≈18 Hz)

Generation method: per-channel sine synthesis with Gaussian noise and optional localized pulses.

---

## 🔬 Feature Engineering

For each sample, features are computed per channel and then concatenated:

- Band powers: delta (1–4 Hz), theta (4–8), alpha (8–13), beta (13–30), gamma (30–80)
- Relative band powers: each band divided by total power
- Time-domain stats: mean, standard deviation, variance
- Spectral shape: spectral centroid and spectral bandwidth

These capture energy distribution, relative composition, stability, and frequency structure.

---

## 🤖 Models

Trained with a stratified split on simulated labeled data:

- Random Forest (ensemble of decision trees)
  - Why: robust to noise, nonlinear patterns, provides feature importance
- SVM (large-margin classifier)
  - Why: strong in high-dimensional spaces, good class boundaries
- Logistic Regression (linear baseline)
  - Why: fast, interpretable baseline and sanity check

The app displays per-model accuracy and prediction confidence at inference time.

---

## 📈 Live Demo (Streamlit)

- Train models with one click
- Generate new EEG sample and predict state
- View real-time plots for frontal/central/occipital channel groups
- See predicted label, confidence, recent history, and top feature importance
- Modern dark theme with neon accents for clarity and impact

---

## 🗂️ Project Structure

```
Brain ML/
├── final_mindwave.py         # Main Streamlit application
├── requirements_final.txt    # Dependencies
├── README.md                 # This guide
└── LICENSE                   # MIT License
```

---

## ❓ FAQ

- Does this use real EEG?  
  No, this version uses simulated EEG for instant, reproducible demos. The pipeline mirrors real EEG workflows.

- Can I plug in real data later?  
  Yes. The feature and model code paths are designed to accept real EEG after preprocessing.

---

## 📜 License

This project is released under the MIT License. See `LICENSE`.

---

Made for research, teaching, and portfolio demos. 🧠✨
