# Deprem_Ai

# 🌍 Real-Time Earthquake Monitoring with Deep Learning

This project implements a **real-time earthquake monitoring and classification system** using official seismic data sources and a **PyTorch-based deep learning model**.

The system continuously collects earthquake data from:

- 🇹🇷 AFAD (Turkey Disaster Authority)
- 🌎 USGS (United States Geological Survey)

Then:

✔ Processes seismic features  
✔ Trains a neural network  
✔ Visualizes training performance  
✔ Performs real-time earthquake event prediction  

---

## 🚀 Features

- Real-time earthquake data streaming
- Automatic fallback (AFAD → USGS)
- Deep learning classification model
- GPU (CUDA) acceleration support
- Live monitoring loop
- Training loss & accuracy visualization
- Modular AI system architecture

---

## 🧠 Model Architecture

Binary classification neural network:

Input Features:

- Underground Wave Energy (synthetic feature)
- Vibration Axis Variation (geospatial dispersion)

Network:

Input(2) → Linear(10) → ReLU → Linear(10) → ReLU → Linear(1)

Loss Function:

BCEWithLogitsLoss

Optimizer:

Adam

Classification Rule:

Magnitude ≥ 4.0 → Significant Seismic Event

---

## 📡 Data Sources

AFAD API  
https://deprem.afad.gov.tr/

USGS Earthquake Feed  
https://earthquake.usgs.gov/

---
