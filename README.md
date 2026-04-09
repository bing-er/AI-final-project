# Military Camouflage Object Detection via Cross-Domain Transfer Learning

**COMP 9130 — Applied Artificial Intelligence | Capstone Project**
**British Columbia Institute of Technology — MSc Applied Computing**

---

## Project Overview

This project investigates whether cross-domain transfer learning from **natural animal camouflage** (COD10K) can improve pixel-level segmentation of **camouflaged military personnel** (ACD1K). We compare a CNN baseline (SINetV2) against a Transformer-based architecture (SegFormer-B2) across three experimental conditions and evaluate cross-environment generalisation across forest, desert/rocky, and snow terrains.

**Research Gap:** Existing camouflage detection research focuses predominantly on animal camouflage. The cross-domain transfer from animal → military camouflage is largely untested, and no published study has evaluated terrain-stratified generalisation on military-specific imagery.

---

## Datasets

| Dataset | Images | Split | Domain | Source |
|---|---|---|---|---|
| COD10K | 10,000 | 6,000 / 4,000 train/test | Natural animal camouflage | [Kaggle](https://www.kaggle.com/datasets/ivanomelchenkoim11/cod10k-dataset) |
| ACD1K | 1,078 | 748 / 330 train/test | Military personnel | [Kaggle](https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k) |
| CAMO | 1,250 | 1,000 / 250 train/test | Mixed natural + artificial | [Official site](https://sites.google.com/view/ltnghia/research/camo) |

> **Data is not included in this repository.** See setup instructions below.

---

## Experimental Design

| Experiment | Architecture | Training Data | Purpose |
|---|---|---|---|
| 1 — CNN Baseline | SINetV2 | ACD1K only | CNN baseline; quantifies performance with military-specific data only |
| 2 — Transfer Learning | SegFormer-B2 | COD10K → ACD1K (two-stage) | Tests cross-domain pretraining benefit: animal → military |
| 3 — Joint Training | SegFormer-B2 | COD10K + CAMO + ACD1K | Tests whether enriched joint distribution improves over sequential transfer |

All three experiments are evaluated on the **same 200-image held-out final test set** (100 ACD1K + 50 COD10K + 50 noise distractors) with terrain-stratified breakdowns (forest, desert/rocky, snow).

---

## Repository Structure

```
AI-final-project/
├── data/                              # Datasets — not tracked by Git (see DATA_DOWNLOAD_INSTRUCTIONS.txt)
│   ├── DATA_DOWNLOAD_INSTRUCTIONS.txt # Step-by-step dataset download guide
│   ├── COD10K-v3/                     # COD10K: 10,000 animal camouflage images
│   ├── dataset-splitM/                # ACD1K: 1,078 military camouflage images
│   └── CAMO-V.1.0-CVIU2019/           # CAMO: 1,250 mixed camouflage images
├── doc/
│   └── 01_EDA_Binger.pdf              # EDA notebook exported as PDF for reference
├── notebooks/
│   ├── 01_EDA_Binger.ipynb            # EDA — class distribution, resolution, mask coverage, normalization constants
│   ├── 02_train_exp1_Yansong.ipynb    # Experiment 1 — SINetV2 baseline    
│   ├── 02_train_exp2_Sepehr.ipynb     # Experiment 2 — SegFormer transfer
│   ├── 02_train_exp3_Binger.ipynb     # Experiment 3 — joint training sweep + final run on A100 
│   └── 03_evaluate_Sepehr.ipynb       # Final evaluation script
├── outputs/
│   └── exp1/
│       ├── final/                     # Final run (lr=6e-5, 60 epochs, early stopped @ ep7) — mIoU=0.6174)
│       ├── sweep_lr1e4/               # Sweep run 1: lr=1e-4, 30 epochs — mIoU=0.5981
│       ├── sweep_lr1e5/               # Sweep run 3: lr=1e-5, 20 epochs — mIoU=0.5865
│       └── sweep_lr6e5/               # Sweep run 2: lr=6e-5, 20 epochs — mIoU=0.5322
│   └── exp2/
│       ├── eval/                     
│       ├── final/                     
│       │   ├── config.json           
│       │   └── history.json          
│       ├── final_lr1e4/               
│       │   ├── config.json
│       │   └── history.json
│       ├── final_lr6e5_50ep/         
│       │   ├── config.json
│       │   └── history.json
│       ├── sweep_lr1e4/               
│       │   ├── config.json
│       │   └── history.json
│       ├── sweep_lr1e5/              
│       │   ├── config.json
│       │   └── history.json
│       └── sweep_lr6e5/               
│           ├── config.json
│           └── history.json
│   └── exp3/
│       ├── final/                     # Final run (lr=6e-5, A100, early stopped @ ep7) — mIoU=0.8717
│       │   ├── config.json            # Training hyperparameters
│       │   └── history.json           # Per-epoch metrics
│       ├── final_lr1e4/               # Final run (lr=1e-4, A100, early stopped @ ep35) — mIoU=0.8712
│       │   ├── config.json            # Training hyperparameters
│       │   └── history.json           # Per-epoch metrics
│       ├── final_lr6e5_50ep/          # Definitive final run (lr=6e-5, A100, early stopped @ ep17) — mIoU=0.8780 ✅
│       │   ├── config.json            # Training hyperparameters
│       │   └── history.json           # Per-epoch metrics
│       ├── sweep_lr1e4/               # Sweep run 1: lr=1e-4, 20 epochs — mIoU=0.8735, @ ep14
│       │   ├── config.json            # Training hyperparameters
│       │   └── history.json           # Per-epoch metrics
│       ├── sweep_lr6e5/               # Sweep run 2: lr=6e-5, 20 epochs — mIoU=0.8805, @ ep17
│       │   ├── config.json            # Training hyperparameters
│       │   └── history.json           # Per-epoch metrics
│       └── sweep_lr1e5/               # Sweep run 3: lr=1e-5, 20 epochs — mIoU=0.8643, @ ep18
│           ├── config.json
│           └── history.json
│   └── figures/                       # EDA figures
├── splits/                            # Fixed split index files — version controlled, generated once with seed=42
│   ├── acd1k_splits.json              # ACD1K: 748 train / 230 val filenames
│   ├── cod10k_splits.json             # COD10K: 5,950 train / 3,950 val filenames
│   ├── hold_out_acd1k.json            # 100 ACD1K hold-out images (from official test partition)
│   ├── hold_out_cod10k.json           # 50 COD10K hold-out images (from official test partition)
│   └── hold_out_noise.json            # 50 NonCAM distractor images (from COD10K train partition)
├── src/
│   ├── __init__.py                    # Package root
│   ├── dataset.py                     # Data loading, augmentation, DataLoader factory for all 3 conditions
│   ├── evaluate.py                    # Evaluation metrics: mIoU, F1/Dice, MAE, FPR
│   ├── generate_splits.py             # Generates all split JSON files — run once before trainin
│   ├── engine_exp1.py 
│   ├── train_exp2.py
│   └── train_exp3.py                     
├── .gitignore                         # Excludes data/, checkpoints, figures, __pycache__
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/bing-er/AI-final-project.git
cd AI-final-project
```

### 2. Install dependencies

```bash
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

---

## Usage

### EDA (Binger)

```bash
# Open in Google Colab or JupyterLab
notebooks/01_EDA_Binger.ipynb
```

### Experiment 1 — SINetV2 CNN Baseline (Yansong)

```bash
# Train SINetV2 on ACD1K only
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
 
 python src/engine_exp1.py \
    --lr 1e-4 --epochs 60 --batch_size 8 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp1

python src/engine_exp1.py --epochs 60 --batch_size 8
```

### Experiment 2 — SegFormer Transfer Learning (Sepehr)

```bash
# Stage 1: Pretrain on COD10K

# Stage 2: Fine-tune on ACD1K

# Evaluate on final test set

```

### Experiment 3 — SegFormer Joint Training (Binger)

```bash
# Train on COD10K + CAMO + ACD1K jointly
python src/train_exp3.py \
    --lr 1e-4 --acd1k_w 8.0 --epochs 20 \
    --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/sweep_lr1e4
    
python src/train_exp3.py \
    --lr 6e-5 --acd1k_w 8.0 --epochs 20 \
    --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/sweep_lr6e5   
    
python src/train_exp3.py \
    --lr 1e-5 --acd1k_w 8.0 --epochs 20 \
    --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/sweep_lr1e5
    
python src/train_exp3.py \
    --lr 6e-05 --acd1k_w 8.0 --epochs 50 \
    --batch_size 16 --accum_steps 1 \
    --data_root data/ --splits_dir splits/ \
    --output_dir outputs/exp3/final

notebooks/02_train_exp3_Binger.ipynb

```


## Team

| Member | Role | Specific Responsibilities                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|---|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Binger Yu | Data & Preprocessing Lead + Experiment 3 Lead | - Generated and version-controlled final test set hold-out index files. Conducted full EDA (`01_EDA_Binger.ipynb`). <br/>- Designed and implemented the complete preprocessing pipeline (`src/dataset.py`): mask binarization, Albumentations augmentation, and `WeightedRandomSampler` for ACD1K oversampling. <br/>- Implemented `src/generate_splits.py` for reproducible splits with fixed random seed across all three experiments. Implemented dataloaders for all three experimental conditions. <br/>- Generated `src/evaluate.py` for final evaluation. <br/>- Led Experiment 3 (joint training): Conducted training exp3 (`02_train_exp3_Binger.ipynb`). Performs hyperparameter tuning, full final training, and final test set evaluation. <br/>- Maintains GitHub repository structure, `README.md`, `requirements.txt`, `.gitignore`, and dataset download instructions. <br/>- Contributes to proposal and report (Background, Dataset, Timeline, Team Roles, Conclusion, Contributions, Acknowledgements). |
| Yansong Jia | Methodology Lead + Experiment 1 Lead | - Owns the overall methodology: defines and documents the experimental design, fixed hyperparameters, augmentation policy, and evaluation protocol. Revises methodology if issues arise during implementation. <br/>- Tracks project progress across all three experiments; facilitates cross-member communication and knowledge transfer. <br/>- Leads Experiment 1: implements and trains SINetV2 on ACD1K (CNN baseline), performs hyperparameter tuning, and evaluates on the final test set. <br/>- Contributes to proposal and report (Background, Research Gap, Methodology, Discussion — Ethical Considerations). <br/>- Prepare PowerPoint.                                                                                                                                                                                                                                                                                                                                                                       |
| Sepehr Mansouri | Experiment 2 Lead + Evaluation Lead | - Leads Experiment 2: implements and trains the SegFormer transfer learning pipeline (COD10K pretraining → ACD1K fine-tuning), performs hyperparameter tuning for both stages, and evaluates on the final test set. <br/>- Develops and maintains all evaluation scripts: mIoU, F1/Dice, MAE, FPR on noise images, terrain-stratified breakdowns, and cross-experiment comparison visualisations. <br/>- Leads system integration and supports debugging across all three pipelines. <br/>- Contributes to proposal and report (Success Criteria, Experiments & Results, Discussion — Analysis of Results and Failure Cases & Limitations). <br/>- Prepare PowerPoint.                                                                                                                                                                                                                                                                                                                                                                                |

---

## Project Status

Last updated: April 2, 2026

### Implementation Progress

| Component                                                                       | Owner          | Status |
|---------------------------------------------------------------------------------|----------------|---|
| Dataset acquisition (COD10K, ACD1K, CAMO)                                       | Binger         | ✅ Complete |
| EDA notebook (`notebooks/01_EDA_Binger.ipynb`)                                  | Binger         | ✅ Complete |
| Preprocessing pipeline (`src/dataset.py`)                                       | Binger         | ✅ Complete |
| Split generator (`src/generate_splits.py`)                                      | Binger         | ✅ Complete |
| Split index files (`splits/`)                                                   | Binger         | ✅ Complete |
| Experiment 3 training (`src/train_exp3.py`)                                     | Binger         | ✅ Complete |
| Experiment 3 notebook (`notebooks/02_train_exp3_Binger.ipynb`)                  | Binger         | ✅ Complete |
| Evaluation script (`src/evaluate.py`)                                           | Binger         | ✅ Complete |
| Experiment 1 — SINetV2 baseline (`src/engine_exp1.py`)                          | Yansong        | ✅ Complete |
| Experiment 1 notebook (`notebooks/02_train_exp1_Yansong.ipynb`)                 | Yansong        | ✅ Complete |
| Experiment 2 — SegFormer transfer (`src/train_exp2.py`)                         | Sepehr         | ✅ Complete |
| Experiment 2 notebook (`notebooks/03_train_exp2_Sepehr.ipynb`)                  | Sepehr         | 🔄 In progress |
| Final evaluation on 200-image hold-out (`src/evaluate.py`)                      | Sepehr         | ✅ Complete |
| Final evaluation on 200-image hold-out (`notebooks/03_evaluate_Sepehr.ipynb`)   | Sepehr         | 🔄 In progress |
| Final report                                                                    | All            | ⏳ In progress |
| PowerPoint                                                                      | Yansong&Sepehr | ⏳ In progress |

### Experiment 3 Results (SegFormer-B2, Joint Training)

| Metric | Value | Target |
|---|---|---|
| val mIoU | **0.8780** | ≥ 0.65 ✅ |
| val F1 | **0.8721** | ≥ 0.75 ✅ |
| val MAE | **0.0338** | lower is better ✅ |

Best checkpoint: `outputs/exp3/final_lr6e5_50ep/best_model.pth`  
Hardware: Google Colab A100 | LR: 6e-5 | Batch: 16 | Early stop: epoch 27

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

### SINetV2 (Experiment 1)
- Two-stage CNN architecture with texture-enhanced modules and group-reversal attention
- Input resolution: 352×352
- Reference: Fan et al., IEEE TPAMI 2022

### SegFormer-B2 (Experiments 2 & 3)
- Hierarchical Vision Transformer encoder (MiT-B2) + lightweight all-MLP decoder
- Initialised from `nvidia/segformer-b2-finetuned-ade-512-512`
- Input resolution: 512×512
- Reference: Xie et al., NeurIPS 2021

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection-over-Union — primary segmentation metric |
| **F1 / Dice** | Harmonic mean of precision and recall on foreground pixels |
| **MAE** | Mean Absolute Error between predicted and ground-truth masks |
| **FPR** | False Positive Rate on 50 noise distractor images |

Terrain-stratified breakdowns (forest, desert/rocky, snow) and cross-experiment comparison visualisations are generated by `src/evaluate.py`.

---

## References

1. Fan et al., "Camouflaged Object Detection," CVPR 2020.
2. Fan et al., "Concealed Object Detection," IEEE TPAMI 2022.
3. Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers," NeurIPS 2021.
4. Haider et al., "Identification of Camouflage Military Individuals with DFAN and SINetV2," *Scientific Reports*, 2025.
5. Le et al., "Anabranch Network for Camouflaged Object Segmentation," *CVIU*, 2019.

---

## License

For academic use only. Datasets are subject to their respective licenses.
