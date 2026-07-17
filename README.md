# 🧠 Reinforcement Learning Based Brain Tumor Detection
### Using MRI Images with Grad-CAM Explainability

---

## 📋 Project Overview

This project combines **Deep Learning (CNN)** and **Reinforcement Learning (DQN)**
to automatically detect, classify, and localize brain tumors in MRI images.

| Component | Technology |
|-----------|-----------|
| Feature Extractor | ResNet50 (Transfer Learning) |
| Classifier | 4-class FC head (Glioma / Meningioma / No Tumor / Pituitary) |
| RL Agent | Deep Q-Network (DQN) with Experience Replay |
| Explainability | Grad-CAM heatmap visualization |
| Framework | PyTorch |

---

## 📁 Folder Structure

```
BrainTumorRL/
│
├── dataset/                      ← Place Kaggle data here
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       └── (same structure)
│
├── models/
│   ├── resnet50_model.py         ← CNN backbone + classifier
│   └── dqn_agent.py              ← DQN Q-network + replay buffer
│
├── environment/
│   └── tumor_env.py              ← Custom RL environment (bounding box)
│
├── training/
│   ├── train_classifier.py       ← 2-stage CNN training
│   └── train_rl_agent.py         ← DQN episode training loop
│
├── explainability/
│   └── grad_cam.py               ← Grad-CAM hooks + visualization
│
├── evaluation/
│   └── metrics.py                ← Accuracy, F1, IoU, Dice
│
├── utils/
│   └── preprocessing.py          ← DataLoader + transforms
│
├── results/                      ← Auto-generated outputs
│   ├── best_cnn.pth
│   ├── best_dqn.pth
│   ├── cnn_training_curves.png
│   ├── rl_training_curves.png
│   ├── confusion_matrix.png
│   ├── classification_report.json
│   └── gradcam_*.png
│
├── main.py                       ← Orchestrator CLI
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Extract into `dataset/` so you have:
```
dataset/Training/glioma/
dataset/Training/meningioma/
dataset/Training/notumor/
dataset/Training/pituitary/
dataset/Testing/...
```

### 3. Train CNN Classifier
```bash
python main.py --mode train_cnn
```
- Stage 1 (10 epochs): Only the FC head is trained (backbone frozen)
- Stage 2 (20 epochs): Full fine-tuning with lower learning rate
- Saves `results/best_cnn.pth` and training curves

### 4. Train RL Agent
```bash
python main.py --mode train_rl
```
- Runs 500 DQN training episodes
- Each episode = one MRI image; agent moves a bounding box
- Saves `results/best_dqn.pth`

### 5. Evaluate
```bash
python main.py --mode evaluate
```
- Prints classification report + confusion matrix
- Computes mean IoU and Dice score

### 6. Single-Image Inference
```bash
python main.py --mode infer --img dataset/Testing/glioma/Te-gl_0010.jpg
```
Output:
```
[Inference] Predicted Class : glioma
[Inference] Confidence      : 94.32%
[Inference] Bounding Box (pixels): (54, 48, 178, 196)
[Inference] ✓ Output saved → results/gradcam_Te-gl_0010.png
```

### 7. Run Web UI (NeuroLoc App)
To run the interactive Streamlit dashboard for tumor detection, classification, and localization:
```bash
streamlit run app.py
```

---


## 🏗 Architecture

```
MRI Image (224×224)
       │
  ResNet50 Backbone  (ImageNet pre-trained, then fine-tuned)
       │
  Global Avg Pool → 2048-d feature vector
       │                    │
  FC Head (4 classes)   RL State: [features | bbox (4)]
       │                    │
  Tumor Classification   DQN Agent
                             │
                         Bounding Box Actions:
                         UP / DOWN / LEFT / RIGHT
                         ZOOM IN / ZOOM OUT / STOP
                             │
                         Grad-CAM Heatmap
```

---

## ⚙ Reward Function

| Condition | Reward |
|-----------|--------|
| STOP + correct class | +5.0 |
| STOP + wrong class | -3.0 |
| Confidence improved | +1.0 |
| Confidence decreased | -1.0 |
| Episode timeout | -2.0 |

---

## 📊 Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score (per class + macro)

### Localization
- **IoU** (Intersection over Union / Jaccard Index)
- **Dice Score** (F1 at pixel level)

---

## 🔬 Grad-CAM Output

Each inference produces a 3-panel figure:
1. **Original MRI**
2. **Grad-CAM Heatmap Overlay** (jet colormap)
3. **Heatmap + RL Bounding Box** (green rectangle)

---

## 📦 Key Libraries

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | CNN, DQN, data loading |
| `opencv-python` | Grad-CAM colormapping, bbox drawing |
| `scikit-learn` | Classification metrics |
| `matplotlib` / `seaborn` | Plots and confusion matrix |
| `gymnasium` | RL environment API compatibility |
| `streamlit` | Interactive web dashboard and visualization |


---

## 📝 Citation

Dataset: Masoud Nickparvar, *Brain Tumor MRI Dataset*, Kaggle, 2021.
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
