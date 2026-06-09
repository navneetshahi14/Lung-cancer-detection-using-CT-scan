# 🫁 Lung Cancer Classification from CT Scans

A deep-learning research pipeline for **3-class CT scan classification** (Benign / Malignant / Normal) built on the IQ-OTHNCCD lung cancer dataset. The project systematically compares single backbones, multi-CNN feature-fusion hybrids, and soft/weighted ensembles under a wide grid of medical-imaging preprocessing configurations.

> **Best result (so far):** weighted ensemble (ResNet50 + VGG16 + EfficientNet-B0 + Inception-V3) with CLAHE + sharpening → **~99.5 % test accuracy** at 30 epochs.

---

## 📌 Table of Contents
- [Highlights](#-highlights)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Architectures](#-architectures)
- [Preprocessing Pipelines](#-preprocessing-pipelines)
- [Training Setup](#-training-setup)
- [Evaluation & Explainability](#-evaluation--explainability)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Utilities](#-utilities)
- [Notebooks](#-notebooks)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## ✨ Highlights

- **Three modelling paradigms** in one repo: single-CNN baselines, multi-CNN feature-fusion hybrids, and two-head ensembles (soft-vote + val-F1-weighted).
- **Composable preprocessing ablation** — 16 configurations mixing CT windowing, CLAHE, histogram equalization, median/Gaussian denoising, sharpening, and minmax/z-score normalization.
- **Class-imbalance aware training** — square-root inverse-frequency class weights inside `CrossEntropyLoss`, plus a `WeightedRandomSampler` available on demand.
- **Reproducible, leak-free splits** — `tesst.py` performs MD5 content hashing across train/val/test to guarantee zero image overlap.
- **Rich metrics & explainability** — accuracy, precision, recall, macro-F1, confusion matrix, multi-class ROC-AUC, **Grad-CAM** heatmaps.
- **Robust training loop** — mixed-precision (AMP), gradient clipping, `ReduceLROnPlateau`, early stopping on validation F1, checkpointing of best model.

---

## 📂 Dataset

| Class | Images |
|-------|-------:|
| Bengin cases   | 120 |
| Malignant cases | 561 |
| Normal cases    | 416 |
| **Total**       | **1,097** |

- **Source:** [IQ-OTHNCCD lung cancer dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) (Kaggle).
- **Splits:** `train / val / test` folders under `lung_ct_split_no_dup/`. Duplicate images (exact MD5 matches) across splits were removed before training.
- **Optional segmentation:** `lung_ct_lung_only/` contains lung-masked versions produced by Otsu-threshold-based segmentation (`utils/segmentation.py`).

> **Class imbalance:** Malignant ≫ Normal ≫ Benign. Class weights are computed automatically per training run.

---

## 🗂 Project Structure

```
final year project/
├── The IQ-OTHNCCD lung cancer dataset/   # raw images (Bengin / Malignant / Normal)
├── dataset/                              # working copy of the dataset
├── lung_ct_split / _no_dup / _sequential_split / _lung_only
│                                         # various split / preprocessing variants
├── models/
│   ├── single_models.py                  # VGG16, ResNet50, EffNet, DenseNet, Inception, ViT, ConvNeXt
│   ├── hybrid_models.py                  # CNN-Transformer / Multi-CNN feature fusion
│   └── ensemble.py                       # SoftVotingEnsemble, WeightedEnsemble
├── utils/
│   ├── dataloader.py                     # ImageFolder loaders + transforms
│   ├── preprocessing.py                  # CT windowing, CLAHE, denoise, sharpen, normalize
│   ├── segmentation.py                   # Otsu-based lung mask
│   ├── balancing.py                      # class weights, FocalLoss, WeightedRandomSampler
│   ├── train.py                          # train_one_epoch, evaluate, train_model, test_model
│   ├── metrics.py                        # full metrics, ROC, confusion, CSV/Excel logging
│   ├── gradcam.py                        # Grad-CAM (activations + gradients hooks)
│   ├── kfold.py                          # Stratified K-Fold driver
│   ├── plotting.py                       # training curve plotting
│   └── load_model.py                     # checkpoint loader
├── results/
│   ├── checkpoints/                      # per-experiment *.pth best checkpoints
│   ├── reports/                          # all CSV/Excel experiment logs
│   ├── roc_auc/, confusion/, gradcam/    # per-experiment plots & heatmaps
│   ├── training_curves/, training_curve/ # loss/acc/F1 over epochs
│   ├── charts/, charts-seaborn/          # model comparison bar charts
│   └── ...
├── SingleModelPipeline.ipynb             # baseline single-CNN training
├── HybridModelPipeline.ipynb             # multi-CNN hybrid training
├── EnsembleModel.ipynb                   # soft + weighted ensembles
├── tesst.py                              # MD5 leakage detector
└── README.md
```

---

## 🧠 Architectures

### 1. Single Backbones (`models/single_models.py`)
| Key | Backbone | Source |
|-----|----------|--------|
| `resnet50` | ResNet-50 | `timm` |
| `efficientnet` / `efficientnet_b4` | EfficientNet B0 / B4 | `timm` |
| `densenet` | DenseNet-121 | `timm` |
| `inception` | Inception-V3 | `timm` |
| `vit` | ViT Base Patch16 224 | `timm` |
| `convnext` | ConvNeXt-Tiny | `timm` |
| `vgg16` | VGG-16 (custom) | `torchvision` |

Helpers: `freeze_backbone()`, `unfreeze_last_layers()`.

### 2. Hybrid Multi-CNN (`models/hybrid_models.py`)
- `CNNTransformerHybrid` — single CNN backbone → 512-dim → classifier (CNN+ViT fusion kept in code, currently CNN-only).
- `MultiCNNTransformerHybrid` — concatenates features from **N CNN backbones** (e.g. ResNet50 + VGG16 + Inception-V3) into a dynamic-size vector, then a `1024 → 512 → num_classes` MLP head with BN, ReLU, and dropout (0.3 / 0.2).

### 3. Ensembles (`models/ensemble.py`)
- `SoftVotingEnsemble` — averages softmax logits.
- `WeightedEnsemble` — weights each model's logits by its validation F1 (or any chosen score).

---

## 🧪 Preprocessing Pipelines

Each experiment selects a subset of the operations below, exposed as a dict in the notebooks:

| Key | Operation |
|-----|-----------|
| `windowing` | CT lung window (center = −600 HU, width = 1500 HU) |
| `clahe`     | Contrast-Limited Adaptive Histogram Equalization |
| `hist_eq`   | Global histogram equalization |
| `gaussian`  | Gaussian blur (5×5) denoise |
| `median`    | Median filter (5×5) denoise |
| `sharpen_flag` | Unsharp-mask-style sharpening kernel |
| `norm_type` | `minmax` (default) or `zscore` |

Augmentations (train only): horizontal flip, rotation ±20°, random resized crop, random affine, color jitter, random erasing.

Named configurations used in the experiments: `baseline`, `windowing`, `clahe`, `window_clahe`, `clahe_median`, `window_clahe_zscore`, `full_medical_pipeline`, `full_gaussian_pipeline`, `clahe_sharpen`, …

---

## 🏋️ Training Setup

- **Framework:** PyTorch + `timm`
- **Device:** CUDA if available
- **Loss:** `CrossEntropyLoss` with class weights = √(total / class_count), normalized so they sum to `num_classes` (see `compute_single_class_weights`)
- **Optimizer:** AdamW (lr = 1e-4, weight_decay = 1e-4)
- **Scheduler:** `ReduceLROnPlateau` on val loss (factor 0.1, patience 3)
- **Mixed precision:** `torch.amp.GradScaler` for forward/backward
- **Gradient clipping:** `clip_grad_norm_(..., 1.0)`
- **Batch size:** 16 (image size 224×224)
- **Epochs:** 30 / 100 (configurable)
- **Early stopping:** patience 10–20 on val F1
- **Checkpointing:** saves `model_state_dict`, `optimizer_state_dict`, val acc/F1/loss whenever val F1 improves

---

## 📊 Evaluation & Explainability

For every experiment the pipeline records:

- **Per-class recall** for Benign / Malignant / Normal
- **Macro / weighted F1**, precision, recall
- **Confusion matrix** (heatmap)
- **Multi-class ROC-AUC** (one-vs-rest, plotted per class)
- **Training curves** (train/val loss, train/val accuracy, val F1)
- **Grayscale sample dumps** for the test batch
- **Grad-CAM overlays** on misclassified / sample cases (target layer auto-picked: `layer4[-1]` for ResNet, `features[-1]` for VGG/DenseNet/EfficientNet)

---

## 🏆 Results

Headline numbers (full grids in `results/reports/*.csv`):

| Model / Pipeline | Test Acc | F1 |
|---|---:|---:|
| ResNet50 baseline (20 ep)                      | 0.886 | 0.895 |
| ResNet50 clahe_median (20 ep)                 | 0.976 | 0.976 |
| ConvNeXt-Tiny window_clahe_zscore (20 ep)     | 0.988 | 0.988 |
| Inception-V3 clahe_median (20 ep)             | 0.988 | 0.988 |
| ResNet50+VGG16+EffNet-B0 hybrid · clahe_median (100 ep) | 0.991 | 0.991 |
| **Weighted ensemble · clahe_sharpen (30 ep)** | **0.9953** | **0.9952** |
| Soft ensemble · clahe_sharpen (30 ep)         | 0.9905 | 0.9905 |

> Ensemble outperforms every individual backbone; CLAHE + (median denoise or sharpening) consistently ranks among the strongest preprocessing combinations.

---

## ▶️ How to Run

1. **Clone the repo and set up the environment**

   ```bash
   git clone <your-fork-url>
   cd "final year project"

   # Recommended Python 3.10+ with CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install timm scikit-learn opencv-python pandas matplotlib seaborn openpyxl tqdm jupyter
   ```

2. **Place the dataset**

   Put the IQ-OTHNCCD images under `The IQ-OTHNCCD lung cancer dataset/{Bengin cases, Malignant cases, Normal cases}/` (already present in this repo). If you regenerate splits, point `tesst.py` and the notebooks at the desired split folder (default: `lung_ct_split_no_dup`).

3. **Verify no data leakage**

   ```bash
   python tesst.py
   ```

4. **Run a pipeline notebook**

   Open and execute the cells of any of:
   - `SingleModelPipeline.ipynb` — train + evaluate a single backbone
   - `HybridModelPipeline.ipynb`   — train + evaluate a multi-CNN hybrid
   - `EnsembleModel.ipynb`         — soft + weighted ensembles over the trained singles

   Toggle model / preprocessing / epoch configurations at the top of each notebook.

5. **Inspect artefacts**

   Every run writes to `results/`:
   - checkpoints: `results/checkpoints/<model>/<exp_id>_best.pth`
   - CSV logs: `results/reports/*.csv`
   - per-experiment ROC, confusion, training curves, Grad-CAM

---

## 🛠 Utilities

| File | Purpose |
|------|---------|
| `utils/preprocessing.py`  | `CTPreprocess` class + `get_transformss()` |
| `utils/segmentation.py`   | Otsu-based lung masking |
| `utils/dataloader.py`     | `get_dataloaders()` / `get_single_dataloaders()` |
| `utils/balancing.py`      | `compute_class_weights`, `compute_single_class_weights`, `FocalLoss`, `get_weighted_sampler` |
| `utils/train.py`          | `train_one_epoch`, `evaluate`, `train_model`, `test_model` |
| `utils/metrics.py`        | `compute_full_metrics`, `plot_roc_auc`, `plot_confusion_matrix`, `save_results_csv`, Excel loggers |
| `utils/gradcam.py`        | `GradCAM` class with forward + full-backward hooks, `save_gradcam_samples` |
| `utils/kfold.py`          | `run_kfold_training()` for stratified K-fold |
| `utils/plotting.py`       | Combined loss / acc / F1 subplot |
| `utils/load_model.py`     | Reload a checkpoint into a fresh model |
| `tesst.py`                | MD5 content-hash leakage check between splits |

---

## 📓 Notebooks

- **`SingleModelPipeline.ipynb`** — trains each supported backbone with a chosen preprocessing config; logs full metrics, confusion, ROC, training curves, grayscale samples.
- **`HybridModelPipeline.ipynb`** — trains the multi-CNN hybrids (2-, 3-, and 4-backbone variants) under multiple preprocessing configs.
- **`EnsembleModel.ipynb`** — loads checkpoints from `SingleModelPipeline.ipynb` runs and evaluates both `SoftVotingEnsemble` and `WeightedEnsemble`.
- **`barChart.ipynb` / `singleBarChart.ipynb`** — post-hoc analysis charts (model comparisons, per-class breakdowns).
- **`Random.ipynb` / `experiments.ipynb` / `ult.ipynb` / `Untitled.ipynb`** — exploratory scratchpads (segmentation verification, K-fold scratch, etc.).

---

## 🧰 Tech Stack

- **Deep learning:** PyTorch, torchvision, timm
- **Vision / imaging:** OpenCV, PIL
- **ML / metrics:** scikit-learn
- **Data / plots:** pandas, numpy, matplotlib, seaborn, openpyxl

---

## ⚠️ Limitations & Future Work

- **Small dataset** (≈ 1,097 images) — the test-set numbers should be interpreted with care; K-fold runs are recommended for final reporting.
- **No external validation cohort** — only the IQ-OTHNCCD dataset is used.
- **No external test set** — adding LIDC-IDRI, LUNA16, or NSCLC-Radiomics would give a much stronger generalization claim.
- **Hybrid classifier is currently CNN-only fusion** (the ViT branch is kept in code but commented out) — re-enabling it is a natural next step.
- **Explainability limited to Grad-CAM** — adding occlusion / SHAP / attention rollout (for ViT) would strengthen interpretability.
- **No deployment artefact** — wrapping the best ensemble in a FastAPI / Streamlit demo would round out the project.

---

## 👤 Author

Final-year project — Lung Cancer Classification from CT Scans.
Built with PyTorch + timm. Datasets belong to their original curators (IQ-OTHNCCD).
