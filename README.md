# 🧠 Slimmable DuckNet for Clinical Image Segmentation

## 🏥 Project Overview

This project explores the creation of a **Slimmable DuckNet**, an adaptive convolutional neural network architecture, for enhanced clinical utility. It is structured into two main goals:

### 🚩 Project Goals

1. **Slim DuckNet Benchmarking**  
   Train a **slimmable version** of the DuckNet architecture on the **Kvasir-SEG** dataset to benchmark its performance vs traditional (non-slimmable) architectures. This shows how DuckNet tolerates aggressive width scaling (i.e., being "slimmed")—key for deployment on resource-constrained devices.

2. **Clinical Deployment via FEES**  
   Transfer the trained Slimmable DuckNet to a **custom-labeled larynx dataset** for a novel medical application: **Fiberoptic Endoscopic Evaluation of Swallowing (FEES)**. The ultimate aim is to deliver **reliable segmentation under dynamic hardware loads** in real-world hospital settings.

---

## 🌟 Why Slimmable Networks?

In clinical environments like hospitals where **GPU resources are shared**, it's crucial to allow **real-time trade-offs between accuracy and compute**. A slimmable model can dynamically adapt its width (and thereby performance) on-the-fly, enabling:

- Full-accuracy mode when resources are ample  
- Compressed fast mode when many exams are running simultaneously  

> ⚖️ _Sometimes a "meh answer" is better than no answer at all._

---

## 📦 Repository Source and Origin

This project was **forked from** [`medical-segmentation-pytorch`](https://github.com/zh320/medical-segmentation-pytorch) to build a PyTorch-compatible variant of **DuckNet**, which was originally implemented in **TensorFlow** ([DUCK-Net Repo](https://github.com/RazvanDu/DUCK-Net)).

We adapted this PyTorch implementation to:
- More closely match the **hyperparameters and architecture** of the original paper
- Enable **slimmable layers** using techniques from [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks)

---

## ✅ Project Milestones & Progress

### Phase 1: Infrastructure & Baselines
- [x] Set up PyTorch environment on SCC and initialized GitHub repo
- [x] Replicated DuckNet performance on Kvasir-SEG
- [x] Integrated slimmable components into DuckNet
- [ ] Adapted training pipeline to support multi-width slimmable training

### Phase 2: Clinical Adaptation to FEES
- [x] Collected and labeled custom endoscopic larynx dataset
- [ ] Applied transfer learning using base (non-slimmable) DuckNet to FEES data
- [ ] Retrained Slimmable DuckNet on FEES after verifying transfer learning performance


---

## 📁 Project Structure

```
.
├── core/                 # Training loops, loss functions, trainers
├── models/               # DuckNet, Slimmable DuckNet, UNet variants
│   └── slimmable/        # Slimmable ops & modules
├── datasets/             # Dataset definitions and loading logic
├── PolypDataset/        # Training datasets (Kvasir, CVC, our custom)
│   ├── [dataset]/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
├── configs/             # Model and training configs (slim, baseline, etc.)
├── utils/               # Metrics, optimizers, transforms, etc.
├── main.py              # Entry point
├── inference.py         # Inference logic
├── train_script.sh      # SCC job launcher
└── environment.yml      # Conda environment

```

---

## 📊 Datasets

- **Public Datasets:**
   - ETIS_LaribPolypDB
   - CVC_ColonDB
   - CVC_ClinicDB
   - Kvasir_SEG
  
  Use:
  ```bash
  python ./PolypDataset/download_dataset.py
  ```

- **Custom Dataset:**
  - ~100 labeled images of the **pharynx/larynx** collected from **FEES procedures**

---

## 🧪 Model Implementation Notes

- Models reside in `./models`
- New models must be defined in their own class and registered in `__init__.py`
- Training flow controlled via `./core` and `main.py`

> ✅ DuckNet has been enhanced to support width multipliers as defined in slimmable networks (e.g., `[0.25, 0.5, 0.75, 1.0]`)

---

## ⚙️ Installation & Setup

```bash
module load miniconda
git clone https://github.com/<your-username>/Slimmable-DuckNet.git
cd Slimmable-DuckNet
conda env create -f environment.yml
conda activate dl_prj_env
```

To train (example):
```bash
python main.py --config slimducknetconfig
```

To run on the [SCC](https://www.bu.edu/tech/support/research/computing-resources/scc/):
```bash
qsub train_script.sh
```
*Note: this repo supports multiple configs depending on which model or size you want to train, so you may want to adapt the --config parameter accordingly.*

---

## 🔗 References

- 🦆 DUCK-Net (TensorFlow): https://github.com/RazvanDu/DUCK-Net  
- 🧠 Base PyTorch fork: https://github.com/zh320/medical-segmentation-pytorch  
- 🧩 Slimmable Neural Networks (Paper): https://arxiv.org/abs/1812.08928  
- 🔧 Slimmable Networks GitHub: https://github.com/JiahuiYu/slimmable_networks  
