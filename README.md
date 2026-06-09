# NeuroRest-AI
A PyTorch-based Multi-Task Deep Learning Network designed to analyze lifestyle and physiological data to simultaneously classify sleep disorders and predict continuous stress metrics.

# 🧠 NeuroRest AI — Intelligent Sleep Apnea & Stress Risk Analyzer

[cite_start]NeuroRest AI is an end-to-end multi-task deep learning platform engineered to evaluate sleep health profiles and predict psychological stress risks using physiological data and behavioral lifestyle metrics[cite: 238, 242]. 

[cite_start]By substituting traditional machine learning heuristics with a custom **PyTorch-based Multi-Task Learning (MTL) Neural Network**, the system solves both multiclass risk classification and continuous target regression inside a single computational pass[cite: 243]. [cite_start]This joint representation approach allows the network to capture shared biometric features between underlying sleep degradation patterns and human stress markers while heavily reducing training complexity[cite: 243, 270].

[cite_start]🌐 **Live Web Application Demo:** [Hosted on Hugging Face Spaces](https://huggingface.co/spaces/ZohaSarwar/NeuroRest-Al) [cite: 253]

---

## 🚀 Engine Architecture & Performance Architecture

- [cite_start]**🧠 Deep Multi-Task Backbone:** Features a shared multi-layer linear configuration ($128 \rightarrow 64 \rightarrow 32$ neural nodes) optimizing mutual feature extractions[cite: 271].
- **📈 Twin Predictive Output Heads:**
  - [cite_start]**Classification Head:** Maps features to a Softmax distribution across three isolated categorical targets (`None`, `Insomnia`, `Sleep Apnea`)[cite: 273].
  - [cite_start]**Regression Head:** Computes real-time continuous scaling for psychological `Stress Level` flags ($3.0 - 8.0$)[cite: 274].
- [cite_start]**⚙️ High-Integrity Preprocessing Pipeline:** Leverages an engineered pipeline utilizing Scikit-Learn's `StandardScaler` to smooth features with divergent numerical domains (e.g., `Age` vs. `Daily Steps`)[cite: 260, 284, 285]. [cite_start]Maps continuous blood pressure vectors dynamically into isolated `BP_Systolic` and `BP_Diastolic` integer tracks[cite: 259].
- [cite_start]**🛡️ Robust Regularization & Data Leakage Prevention:** Outright eliminates data leakage anomalies by completely stripping target evaluation features from the initial input training metrics[cite: 244, 279]. [cite_start]Implements specialized `BatchNorm1d` layers and a $25\%$ `Dropout` step configuration to handle vanishing gradients and suppress overfitting cycles[cite: 244, 272, 285].
- [cite_start]**📊 Modern Optimization Matrix:** Implements an `AdamW` optimizer combined with a `CosineAnnealingLR` learning rate decay scheduler and cross-task objective balancing ($1.0 \times \text{Loss}_{\text{CE}} + 0.5 \times \text{Loss}_{\text{MSE}}$) to stabilize the dual loss optimization tracks[cite: 276, 288, 289].

---

## 📈 Model Performance Benchmarks

[cite_start]Following optimization convergence and early stopping executions, the validated deep learning model recorded the following baseline metrics[cite: 298]:

- [cite_start]**🛏️ Sleep Disorder Classification Accuracy:** `66.67%` (Weighted F1-Score: `0.6519`) [cite: 299]
- [cite_start]**😓 Stress Level Regression Metrics ($R^2$ Score):** `0.8286` (Root Mean Squared Error: `0.7331`) 

---

## 📂 Repository File Structure

[cite_start]Your repository workspace is organized as follows (matching your active file structure)[cite: 265, 295]:
