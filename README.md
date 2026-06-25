# Assessment of Semantic Segmentation Robustness Under Image Degradation in Railway Environments

## 📌 Overview
This project evaluates the robustness of real-time semantic segmentation models under degraded visual conditions in railway environments.

The study focuses on how segmentation performance is affected by:
- Fog
- Rain
- Snow
- Illumination changes

---

## 🎯 Objectives
- Evaluate robustness under image degradations
- Compare real-time segmentation models
- Analyze cross-condition generalization
- Study performance on real and simulated datasets

---

## 🧠 Models Used
- **DDRNet-23** – Best real-world performance
- **SCTNet-B-Seg75** – Efficient and fast
- **SCTNet-B-Seg100** – Best robustness under degradations

---

## 📂 Datasets

### 1. RailSem19 (Real Dataset)
- ~8,500 annotated railway images
- Pixel-wise semantic labels
- Augmented using:
  - Brightness: 0.5, 0.6, 0.7
  - Contrast: 0.6, 0.7

---

### 2. Simulation Dataset
- Generated using dSPACE environment
- Controlled weather conditions:
  - Fog (500m, 200m)
  - Rain
  - Snow (light, medium, heavy)
  - Different lighting conditions

---

## ⚙️ Training Setup
- Optimizer: AdamW
- Learning Rate: 0.0001
- Epochs: 150
- GPU: NVIDIA A100
- Same setup used for all models

---

## 📊 Results

### 🔹 RailSem19 Performance
| Model | mIoU (%) |
|------|--------|
| SCTNet-B-Seg75 | 85.02 |
| SCTNet-B-Seg100 | 87.58 |
| DDRNet-23 | **90.38** |

---

### 🔹 Key Observations
- DDRNet-23 achieved best real-world performance
- SCTNet-B-Seg100 showed strongest robustness
- Performance drops significantly under fog and snow
- Models trained on moderate conditions generalize better

---

## 🖼️ Example Results
(You can add sample images here later)

---

## 🚀 How to Run

### Activate environment
```bash
conda activate /data/pool/qmc-41b/.conda_envs/ddrnet23
```

### Run benchmark
```bash
python benchmark_ddrnet_fps.py \
  --checkpoint <path_to_checkpoint> \
  --num_classes 6 \
  --height 512 \
  --width 1024
```

---

## 📁 Project Structure
```
ddrnet23/
sctnet_b_seg75/
sctnet_b_seg100/
```

---

## ⚠️ Note
- Large files (checkpoints, datasets) are not included
- This repository contains code and evaluation scripts only

---

## 👨‍💻 Author
Vamsi Kumar Yerasani  
Master’s in Information Technology  
TH OWL
