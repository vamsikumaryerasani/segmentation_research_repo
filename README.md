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
- <img width="496" height="225" alt="image" src="https://github.com/user-attachments/assets/b72ead6f-1801-480f-9508-4e925038a47b" />
- **SCTNet-B-Seg75** – Efficient and fast
- **SCTNet-B-Seg100** – Best robustness under degradation.
- <img width="496" height="227" alt="image" src="https://github.com/user-attachments/assets/8e99352f-62b5-4c6c-bfa1-9b363182a9f5" />


---

## 📂 Datasets

### 1. RailSem19 (Real Dataset)
- ~8,500 annotated railway images
- Pixel-wise semantic labels
- Augmented using:
  - Brightness: 0.5, 0.6, 0.7
  - Contrast: 0.6, 0.7
  - <img width="1302" height="173" alt="image" src="https://github.com/user-attachments/assets/f20996ef-b47b-4435-9db5-388784137bbf" />

---

### 2. Simulation Dataset
- Generated using dSPACE environment
- Controlled weather conditions:
  - Fog (500m, 200m)
  - Rain
  - Snow (light, medium, heavy)
  - Different lighting conditions
  - <img width="284" height="141" alt="image" src="https://github.com/user-attachments/assets/83c380b1-10d4-4d8f-bdec-e7ec264d10f9" />
  - <img width="284" height="141" alt="image" src="https://github.com/user-attachments/assets/7e97cc7a-60cc-4b00-8ae4-e71c956ecbc1" />
  - <img width="680" height="335" alt="image" src="https://github.com/user-attachments/assets/8a8ddfa2-2304-4b93-ae1d-3af1a6aec686" />

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
<img width="1235" height="551" alt="image" src="https://github.com/user-attachments/assets/a83224b7-ea75-41e6-9a21-fc4a25a8f75f" />

---
## Qualitative Segmentation Results
<img width="1171" height="198" alt="image" src="https://github.com/user-attachments/assets/4d11aa22-be5e-4846-9c95-9ac3a5f8951a" />



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
