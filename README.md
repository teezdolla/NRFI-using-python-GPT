# NRFI Prediction Enhancements

This repository trains a model to predict whether a run will be scored in the first inning (YRFI/NRFI). The original notebook has been extended with reusable Python scripts.

## Data preparation
- `data_prep.py` downloads Statcast data season by season using **pybaseball** and caches the files in `data_cache/` so repeated executions avoid re-downloading.
- Run `python data_prep.py` to fetch the latest seasons.

## Model training
`train_model.py` now uses the leak-free dataset `final_training_data_leakfree.csv`
to avoid using first-inning information from the same game. Rolling pitcher and
team offense stats are merged, along with days of rest and ballpark factors. Missing
values are imputed with column medians rather than zeros. Hyperparameter tuning is
performed with `GridSearchCV` and the final model is trained with early stopping
and probability calibration via isotonic regression. The tuned model is saved as
`xgboost_yrfi_leakfree_tuned.json` and the calibrator as `isotonic_calibrator.pkl`.

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
3. Generate predictions for today's games (model and calibrator are loaded automatically):
   ```bash
   python predict_today.py --output results.csv --txt-output results.txt
   ```
   The script automatically resolves pitcher IDs using MLB-StatsAPI and merges
   the latest rolling stats for both pitchers and team first-inning offense. Results
   are aggregated for the **full first inning** (top and bottom halves combined).
   Use `--output` to save a CSV and `--txt-output` to write a plain-text table.

### Unified pipeline
`nrfi_full_pipeline.py` exposes sub-commands to perform all steps:
```bash
python nrfi_full_pipeline.py fetch_data
python nrfi_full_pipeline.py train
python nrfi_full_pipeline.py predict_today --output results.csv --txt-output results.txt
```
