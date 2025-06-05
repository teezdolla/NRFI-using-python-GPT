# NRFI Prediction Enhancements

This repository trains a model to predict whether a run will be scored in the first inning (YRFI/NRFI). The original notebook has been extended with reusable Python scripts.

## Data preparation
- `data_prep.py` downloads Statcast data season by season using **pybaseball** and caches the files in `data_cache/` so repeated executions avoid re-downloading.
- Run `python data_prep.py` to fetch the latest seasons.

## Model training
`train_model.py` builds an enhanced training set from `final_training_data.csv` by
merging rolling pitcher and team offense statistics. It then performs hyperparameter
tuning using `GridSearchCV` with **time-series cross-validation**. The best model is
evaluated on an 80/20 chronological split and saved as `xgboost_yrfi_tuned.json`.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
You can run individual scripts or use the unified pipeline:

### Stand-alone scripts
1. Fetch or update Statcast data:
   ```bash
   python data_prep.py
   ```
2. Train the tuned model:
   ```bash
   python train_model.py
   ```
3. Generate predictions for today's games:
   ```bash
   python predict_today.py
   ```
   The script automatically resolves pitcher IDs using MLB-StatsAPI and merges
   the latest rolling stats for both pitchers and team first-inning offense.

### Unified pipeline
`nrfi_full_pipeline.py` exposes sub-commands to perform all steps:
```bash
python nrfi_full_pipeline.py fetch_data
python nrfi_full_pipeline.py train
python nrfi_full_pipeline.py predict_today
```
