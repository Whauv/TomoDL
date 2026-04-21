# TomoDL

TomoDL is a PyTorch + MONAI pipeline for detecting bacterial flagellar motors in cryo-ET tomograms for the BYU Locating Bacterial Flagellar Motors challenge.

It includes:
- 3D patch dataset loading for `.mrc`/`.npy`
- MAE-style 3D self-supervised pretraining
- Dual-head multitask model (voxel segmentation + centroid regression)
- Optuna tuning, TTA, ensemble utilities, ablation runner
- Kaggle-ready `submission.csv` generation

## 1. Project Layout

```
tomodl/
├── data/
├── models/
├── training/
├── evaluation/
├── tuning/
├── ablation/
├── visualization/
├── utils/
├── configs/config.yaml
├── train.py
├── predict.py
└── requirements.txt
```

## 2. Environment Setup

From `C:\Users\prana\OneDrive\Documents\Playground\tomodl`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Data Format (CSV)

TomoDL expects CSV manifests:

- `train.csv` and `val.csv` with columns:
  - `tomo_id` (string)
  - `tomo_path` (absolute or relative path to `.mrc` or `.npy`)
  - `has_motor` (`0` or `1`)
  - `z`, `y`, `x` (centroid coordinates; use zeros for negative samples)
- `test.csv` with columns:
  - `tomo_id`
  - `tomo_path`

Example `train.csv` row:

```csv
tomo_id,tomo_path,has_motor,z,y,x
tomo_001,./data/tomos/tomo_001.mrc,1,120,85,210
```

## 4. Prepare Kaggle Competition Files

1. Download and extract Kaggle competition data into:
   - `./data/kaggle/raw`
   - Optional helper (requires Kaggle API setup):

```powershell
.\scripts\download_kaggle_data.ps1
```
2. Generate TomoDL manifests:

```powershell
python .\data\prepare_kaggle_data.py --kaggle-root .\data\kaggle\raw --output-dir .\data
```

or with helper script:

```powershell
.\scripts\prepare_kaggle_data.ps1 -KaggleRoot .\data\kaggle\raw -OutputDir .\data
```

This creates:
- `./data/train.csv`
- `./data/val.csv`
- `./data/test.csv`

## 5. Alternative: Prepare Hugging Face Mirror Data

You can use:
- `Floppanacci/tomogram-Bacterial-Flagellar-motors-location`

One-command preparation (download + manifest generation):

```powershell
.\scripts\prepare_hf_mirror_data.ps1 -Download
```

or two-step:

```powershell
python .\data\prepare_hf_mirror_data.py --download --hf-root .\data\hf_mirror\raw --output-dir .\data
```

This also creates:
- `./data/train.csv`
- `./data/val.csv`
- `./data/test.csv`

## 6. Configure Paths/Hyperparameters

Edit `configs/config.yaml`:
- set `paths.train_csv`, `paths.val_csv`, `paths.test_csv`
- set `paths.output_dir` and checkpoint paths
- tune patch size, lr, batch size, and loss weights

## 7. Training

### Stage 1 + Stage 2 (recommended default)

```powershell
python train.py --config ./configs/config.yaml --stage both
```

### Only MAE pretraining

```powershell
python train.py --config ./configs/config.yaml --stage pretrain
```

### Only multitask fine-tuning

```powershell
python train.py --config ./configs/config.yaml --stage finetune
```

Outputs:
- pretrain checkpoint at `paths.pretrain_ckpt`
- finetune checkpoint at `paths.finetune_ckpt`

## 8. Hyperparameter Search (Optuna)

```powershell
python train.py --config ./configs/config.yaml --stage finetune --optuna
```

Search space:
- `lr`: `1e-5` to `1e-2` (log)
- `patch_size`: `64 | 80 | 96 | 112`
- `w1_seg`, `w2_reg`: `0.1` to `1.0`

## 9. Ablation Study

```powershell
python train.py --config ./configs/config.yaml --ablation
```

Creates an ablation bar plot in `output_dir`.

## 10. Inference + Kaggle Submission

Use best multitask checkpoint:

```powershell
python predict.py --config ./configs/config.yaml --checkpoint ./outputs/finetune_multitask.pt
```

Generated file:
- `paths.submission_path` (default `./outputs/submission.csv`)
- columns: `tomo_id,x,y,z`

## 11. Recommended End-to-End Execution Order

1. Download and extract competition data to `./data/kaggle/raw`.
2. Run `python .\data\prepare_kaggle_data.py --kaggle-root .\data\kaggle\raw --output-dir .\data`.
3. Update `configs/config.yaml` paths and core hyperparameters.
4. Run `python train.py --config ./configs/config.yaml --stage both`.
5. Optionally run Optuna: `python train.py --config ./configs/config.yaml --stage finetune --optuna`.
6. Run inference: `python predict.py --config ./configs/config.yaml --checkpoint <best_ckpt>`.
7. Upload generated `submission.csv` to Kaggle.

## 12. Notes

- The code uses mixed precision automatically when CUDA is available.
- Dataset normalization and patch extraction are config-driven.
- TTA and ensemble weights are controlled in `config.yaml`.
