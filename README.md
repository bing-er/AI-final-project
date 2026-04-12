# Military Camouflage Object Detection via Cross-Domain Transfer Learning

**COMP 9130 — Applied Artificial Intelligence | Capstone Project**  
**British Columbia Institute of Technology — MSc Applied Computing**

---

## Project Overview

This project investigates whether cross-domain transfer learning from **natural animal camouflage** (COD10K) can improve pixel-level segmentation of **camouflaged military personnel** (ACD1K). We compare a CNN baseline (SINetV2) against a Transformer-based architecture (SegFormer-B2) across three experimental conditions, all evaluated on a fixed 200-image held-out test set.

**Research Question:** Does pretraining on large-scale animal camouflage data transfer meaningfully to the military domain, and does multi-source joint training further improve over sequential transfer?

---

## Key Results

All experiments evaluated on the same **200-image held-out test set** (100 ACD1K + 50 COD10K + 50 noise distractors). Primary metric: ACD1K hold-out subset. Success criteria: mIoU ≥ 0.65 and F1 ≥ 0.75.

| Experiment | Architecture | ACD1K mIoU | ACD1K F1 | ACD1K MAE | mIoU Pass | F1 Pass |
|---|---|---|---|---|---|---|
| 1 — CNN Baseline | SINetV2 | 0.6072 | 0.5391 | 0.1822 | ❌ | ❌ |
| 2 — Transfer Learning | SegFormer-B2 | 0.8592 | 0.8542 | 0.0500 | ✅ | ✅ |
| 3 — Joint Training | SegFormer-B2 | **0.8591** | **0.8546** | **0.0486** | ✅ | ✅ |

> Results from `outputs/expN/eval/eval_results.json`. Experiments 2 and 3 are statistically equivalent on military targets; Exp 3 achieves a substantially lower false-positive rate (8% vs 20%) on noise distractor images.

---

## Datasets

| Dataset | Images | Split | Domain | Source |
|---|---|---|---|---|
| COD10K | 10,000 | 6,000 / 4,000 train/test | Natural animal camouflage | [Kaggle](https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset) |
| ACD1K | 1,078 | 748 / 330 train/test | Military personnel | [Kaggle](https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k) |
| CAMO | 1,250 | 1,000 / 250 train/test | Mixed natural + artificial | [Official site](https://sites.google.com/view/ltnghia/research/camo) |

> **Data is not included in this repository.** See [Setup](#setup) below.

---

## Repository Structure

```
AI-final-project/
├── data/                               # Datasets — not tracked by Git
│   ├── DATA_DOWNLOAD_INSTRUCTIONS.txt  # Step-by-step dataset download guide
│   ├── COD10K-v3/                      # COD10K: 10,000 animal camouflage images
│   ├── dataset-splitM/                 # ACD1K: 1,078 military camouflage images
│   └── CAMO-V.1.0-CVIU2019/            # CAMO: 1,250 mixed camouflage images
├── doc/
│   └── 01_EDA_Binger.pdf               # EDA notebook exported as PDF
├── notebooks/
│   ├── 01_EDA_Binger.ipynb             # EDA — class distribution, resolution, mask coverage
│   ├── 02_train_exp1_Yansong.ipynb     # Experiment 1 — SINetV2 baseline (Colab)
│   ├── 02_train_exp2_Sepehr.ipynb      # Experiment 2 — SegFormer transfer (Colab)
│   ├── 02_train_exp3_Binger.ipynb      # Experiment 3 — joint training (Colab)
│   └── 03_evaluate_Sepehr_Yansong.ipynb# Final hold-out evaluation (Colab)
├── outputs/
│   ├── exp1/
│   │   ├── eval/                       # Hold-out evaluation results (eval_results.json)
│   │   ├── final/                      # ✅ Official Exp1 model — lr=1e-4, best @ epoch 43, val_mIoU=0.6174
│   │   ├── sweep_lr1e4/                # Sweep: lr=1e-4, val_mIoU=0.5981
│   │   ├── sweep_lr6e5/                # Sweep: lr=6e-5, val_mIoU=0.5865
│   │   └── sweep_lr1e5/                # Sweep: lr=1e-5, val_mIoU=0.5322
│   ├── exp2/
│   │   ├── eval/                       # Hold-out evaluation results (eval_results.json)
│   │   ├── stage1/                     # COD10K pretraining — best @ epoch 47, val_mIoU=0.8966
│   │   ├── final/                      # ✅ Official Exp2 model — lr=1e-5, best @ epoch 19, val_mIoU=0.8703
│   │   ├── sweep_s2_lr1e4/             # Stage-2 sweep: lr=1e-4, val_mIoU=0.8652
│   │   ├── sweep_s2_lr1e5/             # Stage-2 sweep: lr=1e-5, val_mIoU=0.8588
│   │   └── sweep_s2_lr5e6/             # Stage-2 sweep: lr=5e-6, val_mIoU=0.8513
│   └── exp3/
│       ├── eval/                       # Hold-out evaluation results (eval_results.json)
│       ├── final/                      # ✅ Official Exp3 model — lr=6e-5, best @ epoch 9, val_mIoU=0.8751
│       │                               #    (Colab checkpoint used for all reported results)
│       ├── final_lr1e4/                # Post-eval run: lr=1e-4, best @ epoch 35, val_mIoU=0.8731
│       ├── final_lr6e5_50ep/           # Post-eval run: lr=6e-5, best @ epoch 17, val_mIoU=0.8780
│       │                               #    (trained after evaluation was complete; not used in report)
│       ├── sweep_lr1e4/                # Sweep: lr=1e-4, val_mIoU=0.8735
│       ├── sweep_lr6e5/                # Sweep: lr=6e-5, val_mIoU=0.8805
│       └── sweep_lr1e5/                # Sweep: lr=1e-5, val_mIoU=0.8643
├── splits/                             # Fixed split indices — version-controlled, seed=42
│   ├── acd1k_splits.json               # ACD1K: 748 train / 230 val filenames
│   ├── cod10k_splits.json              # COD10K: 5,950 train / 3,950 val filenames
│   ├── hold_out_acd1k.json             # 100 ACD1K hold-out images
│   ├── hold_out_cod10k.json            # 50 COD10K hold-out images
│   └── hold_out_noise.json             # 50 NonCAM distractor images
├── src/
│   ├── __init__.py
│   ├── dataset.py                      # Data loading, augmentation, DataLoader factory
│   ├── sinetv2.py                      # SINetV2-inspired encoder-decoder architecture (Exp 1)
│   ├── engine_exp1.py                  # Training engine for Experiment 1 (SINetV2)
│   ├── train_exp2.py                   # Training script for Experiment 2 (SegFormer transfer)
│   ├── train_exp3.py                   # Training script for Experiment 3 (joint training)
│   ├── evaluate.py                     # Hold-out evaluation: mIoU, F1, MAE, FPR
│   └── generate_splits.py              # Generates split JSON files — run once
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/bing-er/AI-final-project.git
cd AI-final-project
```

### 2. Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download datasets

- **COD10K**: Download from [Kaggle](https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset), extract to `data/COD10K-v3/`
- **ACD1K**: Download from [Kaggle](https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k), extract to `data/dataset-splitM/`
- **CAMO**: Download from the [official project page](https://sites.google.com/view/ltnghia/research/camo), extract to `data/CAMO-V.1.0-CVIU2019/`

See `data/DATA_DOWNLOAD_INSTRUCTIONS.txt` for detailed instructions.

### 4. Verify dataset loading

```bash
python src/dataset.py data/
```

All three dataset configurations should print ✅.

> **Note:** All training was conducted on Google Colab Pro (NVIDIA A100). The split files in `splits/` are version-controlled with seed=42 and must not be regenerated — doing so would invalidate the held-out test set.

---

## Usage

### EDA (Binger)

Open `notebooks/01_EDA_Binger.ipynb` in Colab or JupyterLab. No arguments required — paths are configured inside the notebook.

---

### Experiment 1 — SINetV2 CNN Baseline (Yansong)

**Hyperparameter sweeps:**

```bash
python src/engine_exp1.py \
    --lr 1e-4 --epochs 30 --batch_size 8 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp1/sweep_lr1e4

python src/engine_exp1.py \
    --lr 6e-5 --epochs 30 --batch_size 8 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp1/sweep_lr6e5

python src/engine_exp1.py \
    --lr 1e-5 --epochs 30 --batch_size 8 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp1/sweep_lr1e5
```

**Final run** (lr=1e-4 selected from sweep):

```bash
python src/engine_exp1.py \
    --lr 1e-4 --epochs 60 --batch_size 8 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp1/final
```

Best checkpoint saved to `outputs/exp1/final/best_model.pth` — converged at epoch 43, val_mIoU=0.6174.

Colab notebook: `notebooks/02_train_exp1_Yansong.ipynb`

---

### Experiment 2 — SegFormer Transfer Learning (Sepehr)

**Stage 1 — Pretrain on COD10K:**

```bash
python src/train_exp2.py \
    --stage 1 \
    --encoder_lr 1e-4 --head_lr 1e-3 \
    --epochs 50 --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp2/stage1
```

**Stage 2 — Hyperparameter sweeps on ACD1K:**

```bash
python src/train_exp2.py \
    --stage 2 --lr 1e-4 --epochs 20 --batch_size 16 --accum_steps 1 \
    --stage1_weights outputs/exp2/stage1/best_model.pth \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp2/sweep_s2_lr1e4

python src/train_exp2.py \
    --stage 2 --lr 1e-5 --epochs 20 --batch_size 16 --accum_steps 1 \
    --stage1_weights outputs/exp2/stage1/best_model.pth \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp2/sweep_s2_lr1e5

python src/train_exp2.py \
    --stage 2 --lr 5e-6 --epochs 20 --batch_size 16 --accum_steps 1 \
    --stage1_weights outputs/exp2/stage1/best_model.pth \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp2/sweep_s2_lr5e6
```

**Final run** (lr=1e-5 selected from sweep):

```bash
python src/train_exp2.py \
    --stage 2 --lr 1e-5 --epochs 50 --batch_size 16 --accum_steps 1 \
    --stage1_weights outputs/exp2/stage1/best_model.pth \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp2/final
```

Best checkpoint saved to `outputs/exp2/final/best_model.pth` — best at epoch 19, val_mIoU=0.8703, training stopped at epoch 28.

Colab notebook: `notebooks/02_train_exp2_Sepehr.ipynb`

---

### Experiment 3 — SegFormer Joint Training (Binger)

**Hyperparameter sweeps:**

```bash
python src/train_exp3.py \
    --lr 1e-4 --acd1k_w 8.0 --epochs 20 --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/sweep_lr1e4

python src/train_exp3.py \
    --lr 6e-5 --acd1k_w 8.0 --epochs 20 --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/sweep_lr6e5

python src/train_exp3.py \
    --lr 1e-5 --acd1k_w 8.0 --epochs 20 --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/sweep_lr1e5
```

**Final run** (lr=6e-5 selected from sweep):

```bash
python src/train_exp3.py \
    --lr 6e-5 --acd1k_w 8.0 --epochs 50 --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/final
```

Best checkpoint saved to `outputs/exp3/final/best_model.pth` — converged at epoch 9, val_mIoU=0.8751. **This is the checkpoint used for all reported evaluation results.**

Colab notebook: `notebooks/02_train_exp3_Binger.ipynb`

---

### Evaluation — Hold-Out Test Set (Sepehr & Yansong)

Runs inference on the fixed 200-image held-out test set (100 ACD1K + 50 COD10K + 50 noise distractors) and writes `eval_results.json`.

**Experiment 2:**

```bash
python src/evaluate.py \
    --checkpoint outputs/exp2/final/best_model.pth \
    --data_root data/ \
    --splits_dir splits/ \
    --output_dir outputs/exp2/eval
```

**Experiment 3:**

```bash
python src/evaluate.py \
    --checkpoint outputs/exp3/final/best_model.pth \
    --data_root data/ \
    --splits_dir splits/ \
    --output_dir outputs/exp3/eval
```

> `evaluate.py` is designed for SegFormer-B2 checkpoints (Experiments 2 and 3). Experiment 1 results are already saved in `outputs/exp1/eval/eval_results.json`.

Expected output (`eval_results.json`):

```
Subset    N    mIoU    F1      MAE
ACD1K    100   0.8591  0.8546  0.0486   ← primary metric
COD10K    50   0.8654  0.8056  0.0185
Noise     50   0.9149  0.8400  0.0102
```

Colab notebook: `notebooks/03_evaluate_Sepehr_Yansong.ipynb`

---

## Reproducing Results End-to-End

To reproduce all reported results from a fresh environment:

1. Complete [Setup](#setup) (install dependencies, download datasets)
2. Verify splits are unchanged: `splits/hold_out_acd1k.json`, `hold_out_cod10k.json`, `hold_out_noise.json` must match the committed versions (do **not** re-run `generate_splits.py`)
3. Train each experiment using the **Final run** commands above
4. Run evaluation using the **Evaluation** commands above
5. Results will be written to `outputs/expN/eval/eval_results.json`

> All random seeds are fixed at 42 throughout training and split generation. GPU results may show minor floating-point variation across hardware.

---

## Normalization Constants

Computed from training sets (values normalized to [0, 1]):

| Dataset | Mean (R, G, B) | Std (R, G, B) |
|---|---|---|
| COD10K | (0.407, 0.424, 0.340) | (0.208, 0.198, 0.194) |
| ACD1K | (0.411, 0.405, 0.327) | (0.196, 0.196, 0.184) |
| CAMO | (0.479, 0.461, 0.360) | (0.197, 0.195, 0.185) |
| Joint | (0.432, 0.430, 0.342) | (0.200, 0.196, 0.188) |

---

## Models

### SINetV2-inspired (Experiment 1)
- Lightweight encoder-decoder with depthwise separable convolutions and squeeze-and-excitation channel attention
- Input resolution: 352×352
- ~1.2M parameters
- Reference: Fan et al., "Concealed Object Detection," IEEE TPAMI 2022

### SegFormer-B2 (Experiments 2 & 3)
- Hierarchical Mix Transformer (MiT-B2) encoder + lightweight all-MLP decoder
- Initialised from `nvidia/segformer-b2-finetuned-ade-512-512`
- Input resolution: 512×512
- Reference: Xie et al., "SegFormer," NeurIPS 2021

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection-over-Union averaged across foreground and background classes |
| **F1 / Dice** | Harmonic mean of precision and recall on foreground pixels |
| **MAE** | Mean Absolute Error between predicted probability map and binary ground-truth mask |
| **FPR** | False Positive Rate on 50 noise distractor images (threshold: >1% predicted foreground) |

---

## Team

| Member | Role | Contributions |
|---|---|---|
| **Binger Yu** | Data & Preprocessing Lead + Experiment 3 Lead | EDA (`01_EDA_Binger.ipynb`); preprocessing pipeline (`src/dataset.py`) including mask binarization, Albumentations augmentation, and `WeightedRandomSampler`; reproducible split generation (`src/generate_splits.py`); DataLoader factory for all three experimental conditions; Experiment 3 training (`src/train_exp3.py`, `02_train_exp3_Binger.ipynb`); GitHub repository setup, README, `requirements.txt`, `.gitignore`; report sections: Dataset, Conclusion, Contributions, Acknowledgements |
| **Yansong Jia** | Methodology Lead + Experiment 1 Lead | SINetV2 architecture (`src/sinetv2.py`); Experiment 1 training engine (`src/engine_exp1.py`); overall experimental design and methodology review; hyperparameter tuning for Exp 1; report sections: Introduction, Related Work, Methodology, Discussion (co-author) |
| **Sepehr Mansouri** | Experiment 2 Lead + Evaluation Lead | SegFormer-B2 transfer learning pipeline (`src/train_exp2.py`); evaluation framework (`src/evaluate.py`); hold-out test set evaluation and visualisations (`03_evaluate_Sepehr_Yansong.ipynb`); hyperparameter sweeps for Exp 2; report sections: Abstract, Experiments & Results, Discussion (co-author) |

---

## Project Status

| Component | Owner | Status |
|---|---|---|
| Dataset acquisition (COD10K, ACD1K, CAMO) | Binger | ✅ Complete |
| EDA notebook | Binger | ✅ Complete |
| Preprocessing pipeline (`src/dataset.py`) | Binger | ✅ Complete |
| Split generator + index files (`splits/`) | Binger | ✅ Complete |
| SINetV2 architecture (`src/sinetv2.py`) | Yansong | ✅ Complete |
| Experiment 1 training (`src/engine_exp1.py`) | Yansong | ✅ Complete |
| Experiment 2 training (`src/train_exp2.py`) | Sepehr | ✅ Complete |
| Experiment 3 training (`src/train_exp3.py`) | Binger | ✅ Complete |
| Evaluation script (`src/evaluate.py`) | Sepehr | ✅ Complete |
| Hold-out evaluation — all 3 experiments | Sepehr & Yansong | ✅ Complete |
| Final report | All | ✅ Complete |
| PowerPoint presentation | Yansong & Sepehr | ⏳ In progress |

---

## References

1. Fan et al., "Camouflaged Object Detection," CVPR 2020.
2. Fan et al., "Concealed Object Detection," IEEE TPAMI 2022.
3. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers," NeurIPS 2021.
4. Haider & Raza, "Assessment of Camouflage in Heterogeneous Environments through Deep Learning," *Engineering Applications of AI*, 2025.
5. Le et al., "Anabranch Network for Camouflaged Object Segmentation," *CVIU*, 2019.
6. Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations," *Information*, 2020.

---

## License

For academic use only. Datasets are subject to their respective licenses.
