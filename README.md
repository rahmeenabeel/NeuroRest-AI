# 🧠 NeuroRest AI: Intelligent Sleep Apnea & Stress Risk Analyzer  

NeuroRest AI is an end-to-end deep learning system designed to analyze sleep health and predict stress levels using physiological and lifestyle data. It leverages a **PyTorch-based Multi-Task Neural Network** to perform both classification and regression simultaneously, enabling a more comprehensive understanding of user health.

---

## 🎯 Overview  

The system predicts:  
- 🛌 **Sleep Disorders** (None, Insomnia, Sleep Apnea)  
- 📈 **Stress Levels** (Continuous Regression Output)  

Unlike traditional single-task models, NeuroRest AI uses a **shared representation learning approach**, allowing it to capture deeper relationships between health indicators while improving efficiency.

---

## 🚀 Key Features  

### 🧠 Multi-Task Learning Architecture  
- Shared neural backbone with dual output heads  
- Simultaneous classification and regression  

### ⚙️ Robust Data Pipeline  
- Feature engineering (e.g., BP split into systolic/diastolic)  
- Label encoding for categorical variables  
- StandardScaler normalization  

### ✅ Improved Model Reliability  
- Eliminated data leakage  
- Batch Normalization for stable training  
- Dropout (25%) to prevent overfitting  

### 📊 Modern Training Strategy  
- Optimizer: AdamW  
- Scheduler: Cosine Annealing  
- Balanced multi-task loss  

### 🌐 Interactive Web Interface  
- Built using Gradio  
- Real-time predictions and health insights  

### ☁️ Deployment  
- Hosted on Hugging Face Spaces  

👉 **Live Demo:**  
https://huggingface.co/spaces/rahmeenabeel/NeuroRest-AI

---

## 🧱 System Architecture  
Data Layer → Preprocessing → Multi-Task Neural Network → Gradio Interface


- **Input:** 11 health & lifestyle features  
- **Backbone:** Fully Connected Layers (128 → 64 → 32)  
- **Outputs:**  
  - Classification Head (Sleep Disorder)  
  - Regression Head (Stress Level)  

---

## 📊 Model Performance  

- **Classification Accuracy:** 66.67%  
- **R² Score (Stress Prediction):** 0.8286  
- **RMSE:** 0.7331  

---

## ⚠️ Challenges & Improvements  

- Prevented data leakage by removing target variables from inputs  
- Improved convergence using feature scaling and BatchNorm  
- Balanced multi-task loss for stable joint learning  

---

## 🛠️ Tech Stack  

- **Python, PyTorch** – Model development  
- **Scikit-learn, Pandas, NumPy** – Data processing  
- **Gradio** – Interface  
- **Hugging Face Spaces** – Deployment  

---
